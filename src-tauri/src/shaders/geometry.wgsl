struct GeometryUniforms {
    inv_m00: f32,
    inv_m01: f32,
    inv_m02: f32,
    inv_m10: f32,
    inv_m11: f32,
    inv_m12: f32,
    inv_m20: f32,
    inv_m21: f32,
    inv_m22: f32,
    cx: f32,
    cy: f32,
    half_diagonal: f32,
    max_radius_sq_inv: f32,
    auto_crop_scale: f32,
    k_distortion: f32,
    lk1: f32,
    lk2: f32,
    lk3: f32,
    lens_dist_amt: f32,
    tca_vr: f32,
    tca_vb: f32,
    lens_model: u32,
    has_lens_correction: u32,
    has_tca: u32,
    src_width: u32,
    src_height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: GeometryUniforms;

fn bilinear_sample(tex: texture_2d<f32>, x: f32, y: f32) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(tex));
    let clamped_x = clamp(x, 0.0, dims.x - 1.0);
    let clamped_y = clamp(y, 0.0, dims.y - 1.0);

    let x0 = u32(floor(clamped_x));
    let y0 = u32(floor(clamped_y));
    let wx = clamped_x - f32(x0);
    let wy = clamped_y - f32(y0);

    let x1 = min(x0 + 1u, u32(dims.x) - 1u);
    let y1 = min(y0 + 1u, u32(dims.y) - 1u);

    let p00 = textureLoad(tex, vec2<u32>(x0, y0), 0);
    let p10 = textureLoad(tex, vec2<u32>(x1, y0), 0);
    let p01 = textureLoad(tex, vec2<u32>(x0, y1), 0);
    let p11 = textureLoad(tex, vec2<u32>(x1, y1), 0);

    let top = mix(p00, p10, wx);
    let bot = mix(p01, p11, wx);
    return mix(top, bot, wy);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }

    let x_f = f32(id.x);
    let y_f = f32(id.y);

    let result_x = params.inv_m00 * x_f + params.inv_m01 * y_f + params.inv_m02;
    let result_y = params.inv_m10 * x_f + params.inv_m11 * y_f + params.inv_m12;
    let result_z = params.inv_m20 * x_f + params.inv_m21 * y_f + params.inv_m22;

    if (abs(result_z) < 1e-6) {
        textureStore(output_texture, id.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let inv_z = 1.0 / result_z;
    var src_x = result_x * inv_z;
    var src_y = result_y * inv_z;

    if (params.auto_crop_scale > 1.0) {
        src_x = params.cx + (src_x - params.cx) / params.auto_crop_scale;
        src_y = params.cy + (src_y - params.cy) / params.auto_crop_scale;
    }

    if (params.has_lens_correction > 0u) {
        let dx = src_x - params.cx;
        let dy = src_y - params.cy;
        let ru2 = dx * dx + dy * dy;
        let ru = sqrt(ru2);

        if (ru > 1e-6) {
            let ru_norm = ru / params.half_diagonal;
            let ru_norm2 = ru_norm * ru_norm;

            var rd_norm: f32;
            if (params.lens_model == 1u) {
                let a = params.lk1;
                let b = params.lk2;
                let c = params.lk3;
                let d = 1.0 - a - b - c;
                rd_norm = ru_norm * (a * ru_norm2 * ru_norm + b * ru_norm2 + c * ru_norm + d);
            } else {
                rd_norm = ru_norm * (1.0 + params.lk1 * ru_norm2 + params.lk2 * (ru_norm2 * ru_norm2) + params.lk3 * (ru_norm2 * ru_norm2 * ru_norm2));
            }

            let effective_r_norm = ru_norm + (rd_norm - ru_norm) * params.lens_dist_amt;
            let scale = effective_r_norm / ru_norm;

            src_x = params.cx + dx * scale;
            src_y = params.cy + dy * scale;
        }
    }

    if (abs(params.k_distortion) > 1e-5) {
        let dx = src_x - params.cx;
        let dy = src_y - params.cy;
        let r2_norm = (dx * dx + dy * dy) * params.max_radius_sq_inv;
        let f = 1.0 + params.k_distortion * r2_norm;
        src_x = params.cx + dx * f;
        src_y = params.cy + dy * f;
    }

    if (params.has_tca > 0u) {
        let rx = params.cx + (src_x - params.cx) * params.tca_vr;
        let ry = params.cy + (src_y - params.cy) * params.tca_vr;
        let bx = params.cx + (src_x - params.cx) * params.tca_vb;
        let by = params.cy + (src_y - params.cy) * params.tca_vb;

        let r = bilinear_sample(input_texture, rx, ry).r;
        let g = bilinear_sample(input_texture, src_x, src_y).g;
        let b = bilinear_sample(input_texture, bx, by).b;

        textureStore(output_texture, id.xy, vec4<f32>(r, g, b, 1.0));
    } else {
        let color = bilinear_sample(input_texture, src_x, src_y);
        textureStore(output_texture, id.xy, vec4<f32>(color.rgb, 1.0));
    }
}
