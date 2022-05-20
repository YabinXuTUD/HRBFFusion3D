 #version 330 core

//input is global model vbo
layout (location = 0) in vec4 vPosition;
layout (location = 1) in vec4 vColorTime;
layout (location = 2) in vec4 vNormRad;
layout (location = 3) in vec4 curv_map_max;
layout (location = 4) in vec4 curv_map_min;

//output is four textures
out vec4 vPosition0;
out vec4 vColorTime0;
out vec4 vNormRad0;
out vec4 curv_map_max0;
out vec4 curv_map_min0;

flat out int vertexId;

uniform mat4 t_inv;
uniform vec4 cam; //cx, cy, fx, fy
uniform float cols;
uniform float rows;
uniform float maxDepth;
uniform int time;
uniform int timeDelta;
uniform int insertSubmap;
uniform int indexSubmap;

uniform sampler2D KeyFrameIDMap;
uniform float KeyFrameIDDimen;

uniform float curvature_valid_threshold;

void main()
{
    //Pose inverse, set point to local frame and then project
    vec4 vPosHome = t_inv * vec4(vPosition.xyz, 1.0); 
    
    float x = 0;
    float y = 0;
    uint index_submap = uint(vColorTime.y);
    float halfPixel = 0.5 / KeyFrameIDDimen;
    float active = float(textureLod(KeyFrameIDMap, vec2(float(index_submap) / KeyFrameIDDimen + halfPixel, 0.5), 0.0));
    //set inactive here    
    if(vPosHome.z > maxDepth || vPosHome.z < 0 || active == 0.0  //|| time - vColorTime.w > 200 //vColorTime.y <  indexSubmap ||
    )
    {
        x = -10;
        y = -10;
        vertexId = 0;
    }
    else
    {
        x = ((((cam.z * vPosHome.x) / vPosHome.z) + cam.x) - (cols * 0.5)) / (cols * 0.5);
        y = ((((cam.w * vPosHome.y) / vPosHome.z) + cam.y) - (rows * 0.5)) / (rows * 0.5);
        vertexId = gl_VertexID;
    }
    
    gl_Position = vec4(x, y, vPosHome.z / maxDepth, 1.0);

    vPosition0 = vec4(vPosHome.xyz, vPosition.w);                          
    vColorTime0 = vColorTime;
    vNormRad0 = vec4(normalize(mat3(t_inv) * vNormRad.xyz), vNormRad.w);
    curv_map_max0 = curv_map_max;
    curv_map_min0 = curv_map_min;
}
