
#version 330 core

layout(points) in;
layout(points, max_vertices = 1) out;

in vec4 vPosition[];
in vec4 vColor[];
in vec4 vNormRad[];
in vec4 curv_map_max[];
in vec4 curv_map_min[];

flat in float zVal[];

out vec4 vPosition0;
out vec4 vColor0;
out vec4 vNormRad0;
out vec4 curv_map_max0;
out vec4 curv_map_min0;

//out float valid;
void main()
{ 
    if(zVal[0] > 0.0)
    {
        vPosition0 = vPosition[0];
        vColor0 = vColor[0];
        vColor0.y = 0; //Unused
        vColor0.z = 1; //This sets the vertex's initialization time
        vNormRad0 = vNormRad[0];

        curv_map_max0 = curv_map_max[0];
        curv_map_min0 = curv_map_min[0];
        EmitVertex();
        EndPrimitive();
    }
    
}
