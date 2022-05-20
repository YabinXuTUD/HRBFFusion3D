#version 330 core

in vec2 texcoord;

out vec4 FragColor;

uniform sampler2D NormalOptSampler;
uniform sampler2D VertexSampler;

uniform float cols;
uniform float rows;

void main()
{
    FragColor = texture(NormalOptSampler, texcoord.xy);
}
