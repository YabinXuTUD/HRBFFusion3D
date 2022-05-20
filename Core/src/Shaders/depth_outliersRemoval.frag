
#version 330 core

in vec2 texcoord;

out float FragColor;

uniform sampler2D rsampler;
uniform sampler2D modelsampler;

uniform float minVal;
uniform float maxVal;

void main()
{
    float raw_depth_value = float(texture(rsampler, texcoord.xy));
    vec4 vModelPose = texture(modelsampler, texcoord.xy);

    if(vModelPose.z == 0)
         FragColor = raw_depth_value;
    else if(raw_depth_value - vModelPose.z > 0.05 || raw_depth_value - vModelPose.z < -0.05){
             FragColor = 0;
        }else{
             FragColor = (raw_depth_value + vModelPose.w * vModelPose.z) / (1 + vModelPose.w);
        }
}
