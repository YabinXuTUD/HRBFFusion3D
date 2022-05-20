
#version 330 core

layout(points) in;
layout(points, max_vertices = 1) out;

in vec4 vPosition[];
in vec4 vColor[];
in vec4 vNormRad[];
in vec4 curv_map_max[];
in vec4 curv_map_min[];

flat in int valid[];

out vec4 vPosition0;
out vec4 vColor0;
out vec4 vNormRad0;
out vec4 curv_map_max0;
out vec4 curv_map_min0;

void main()
{ 
    if(valid[0] > 0)
    {
        vPosition0 = vPosition[0];
        vColor0 = vColor[0];
        vColor0.y = 0;            //use this as the submap index;
        vColor0.z = 1;            //This sets the vertex's initialisation time
        vNormRad0 = vNormRad[0];

        curv_map_max0 = curv_map_max[0];
        curv_map_min0 = curv_map_min[0];
        EmitVertex();
        EndPrimitive();
    }
    
}
