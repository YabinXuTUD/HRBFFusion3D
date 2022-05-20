//This is for hrbf fusion between points, firstly to build up a local hrbf field
//for each input frame point

//kernel fusntion attributes,with this you could build up local function
//phi value

float getWeight(float d2, float support)
{
    float T2 = support * support;
    if(d2 > T2)
        return 0;
    float r = sqrt(d2 / T2);
    float s = 1-r;
    float s4 = s*s*s*s;
    return s4*(4.0*r+1.0);
}
//

////phi derivative
vec3 getWeightD(float vx, float vy, float vz, float d2, float support)
{
    float T2 = support*support;
    if(d2 > T2 || d2 == 0.0)
    {
        return vec3(0.0,0.0,0.0);
    }
    float invT2 = 1.0/T2;
    float r = sqrt(d2*invT2);

    float s = 1.0-r;
    float s3 = s*s*s;
    float t = -20*s3*invT2;
    return vec3(float(vx*t),float(vy*t),float(vz*t));
}

//phi hessian matrix
void getWeightH(float vx, float vy, float vz, float d2, float support, inout float h[9])
{
    float T2 = support*support;
    if(d2 > T2)
    {
        h[0] = h[1] = h[2] = h[3] = h[4] = h[5] = h[6] = h[7] = h[8] = 0.0f;
        return;
    }
    if(d2 == 0.0f)
    {
        h[0] = h[4] = h[8] = -20.0/T2;
        h[1] = h[2] = h[3] = h[5] = h[6] = h[7] = 0.0f;
        return;
    }
    float r = sqrt(d2/T2);
    float s = 1.0-r;
    float s2 = s*s;

    float t1 = 20.0*s2/(T2*T2*r);
    float t2 = -r*s*T2;
    float vx2 = vx*vx;
    float vy2 = vy*vy;
    float vz2 = vz*vz;
    h[0] = t1*(3.0*vx2+t2);
    h[1] = t1*3.0*vx*vy;
    h[2] = t1*3.0*vx*vz;
    h[3] = h[1];//t1*3.0*vx*vy;
    h[4] = t1*(3.0*vy2+t2);
    h[5] = t1*3.0*vy*vz;
    h[6] = h[2];//t1*3.0*vx*vz;
    h[7] = h[5];//t1*3.0*vy*vz;
    h[8] = t1*(3.0*vz2+t2);
}

//third order derivative of phi for curvature estimation
void getWeightT(float vx, float vy, float vz, float d2, float support,inout float t[27])
{
    float T2 = support*support;
    if (d2 > T2 || d2 == 0.0f) //d2 = 0 maybe not right
    {
        for (int i = 0; i < 27; i++)
            t[i] = 0.0f;
        return;
    }
    float r = sqrt(d2 / T2);
    float s = 1.0 - r;
    float s2 = r - 2 + 1 / r;
    float s3 = 60 / (T2*T2);
    float s4 = 1 / (r*r);
    float partial_rx = vx / (T2*r);
    float partial_ry = vy / (T2*r);
    float partial_rz = vz / (T2*r);
    t[0] = s3*(T2*s*s*partial_rx + 2 * vx * s2 + vx * vx * (partial_rx - s4 * partial_rx)); 
    t[1] = s3 * vy * ((partial_rx - s4 * partial_rx) * vx + s2); 
    t[2] = s3 * vz * ((partial_rx - s4 * partial_rx) * vx + s2); 

    t[3] = s3 * (T2*s*s*partial_ry + vx*vx*(partial_ry - s4 * partial_ry)); 
    t[4] = s3 * vx * (( partial_ry - s4 * partial_ry) * vy + s2); 
    t[5] = s3*vx*vz*(partial_ry - s4 * partial_ry); 

    t[6] = s3 * (T2*s*s*partial_rz + vx*vx*(partial_rz - s4 * partial_rz)); 
    t[7] = s3 * vx * vy * (partial_rz - s4 * partial_rz);  
    t[8] = s3 * vx * ((partial_rz - s4 * partial_rz) * vz + s2);

    t[9] = t[1]; 
    t[10] = s3 * (T2 * s * s * partial_rx + vy * vy * (partial_rx - s4 * partial_rx));
    t[11] = s3*vy*vz*(partial_rx - s4*partial_rx); 

    t[12] = t[4];
    t[13]= s3*(T2*s*s*partial_ry + 2 * vy*s2 + vy*vy*(partial_ry - s4*partial_ry)); 
    t[14] = s3*vz*((partial_ry - s4*partial_ry)*vy + s2);

    t[15] = t[7]; 
    t[16] = s3*(T2*s*s*partial_rz + vy*vy*(partial_rz - s4*partial_rz)); 
    t[17] = s3*vy*((partial_rz - s4*partial_rz)*vz + s2);

    t[18] = t[2];
    t[19] = t[11]; 
    t[20]= s3*(T2*s*s*partial_rx + vz*vz*(partial_rx - s4*partial_rx)); 

    t[21] = t[5];  
    t[22] = t[14];  
    t[23] = s3*(T2*s*s*partial_ry + vz*vz*(partial_ry - s4*partial_ry));

    t[24] = t[8]; 
    t[25] = t[17];
    t[26]= s3*(T2*s*s*partial_rz + 2 * vz*s2 + vz*vz*(partial_rz - s4*partial_rz));
}

float hrbfvalue(vec3 poslocal, vec4 vertConf_neighbor[100], vec4 normRad_neighbor[100], int index_N, inout int number_support_points)
{
    float value = 0;
    number_support_points = 0;
    for(int i = 0; i < index_N; i++)
    {
        vec3 sol = 10.0 * normRad_neighbor[i].xyz;
        float vx = poslocal.x - vertConf_neighbor[i].x;
        float vy = poslocal.y - vertConf_neighbor[i].y;
        float vz = poslocal.z - vertConf_neighbor[i].z;
        float dist2 = vx*vx + vy*vy + vz*vz;
        float support = normRad_neighbor[i].w;
        if(support * support < dist2)
            continue;
        vec3 g = getWeightD(vx, vy, vz, dist2, normRad_neighbor[i].w);
        value-= g.x*sol.x + g.y*sol.y + g.z*sol.z;
        number_support_points++;
    }
    return value;
}

vec3 hrbfgradient(vec3 poslocal, vec4 vertConf_neighbor[100], vec4 normRad_neighbor[100], int index_N)
{
    vec3 grad = vec3(0.0,0.0,0.0);

    for(int i = 0; i < index_N; i++)
    {
        vec3 sol = 10.0 * normRad_neighbor[i].xyz;
        float vx = poslocal.x - vertConf_neighbor[i].x;
        float vy = poslocal.y - vertConf_neighbor[i].y;
        float vz = poslocal.z - vertConf_neighbor[i].z;
        float dist2 = vx*vx + vy*vy + vz*vz;
        float weighH[9];
        getWeightH(vx, vy, vz, dist2, normRad_neighbor[i].w, weighH);

        grad.x -= sol.x * weighH[0] + sol.y * weighH[1] + sol.z * weighH[2];
        grad.y -= sol.x * weighH[3] + sol.y * weighH[4] + sol.z * weighH[5];
        grad.z -= sol.x * weighH[6] + sol.y * weighH[7] + sol.z * weighH[8];
    }
    return grad;
}

void hrbfHessianMatrix(inout float g[9], vec3 poslocal, vec4 vertConf_neighbor[100], vec4 normRad_neighbor[100], int index_N)
{
    g[0] = g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = g[7] = g[8] = 0.0f;
    for(int i = 0; i < index_N; i++)
    {
        vec3 sol = 10.0 * normRad_neighbor[i].xyz;
        float vx = poslocal.x - vertConf_neighbor[i].x;
        float vy = poslocal.y - vertConf_neighbor[i].y;
        float vz = poslocal.z - vertConf_neighbor[i].z;
        float dist2 = vx*vx + vy*vy + vz*vz;
        float hw[27];
        getWeightT(vx, vy, vz, dist2, normRad_neighbor[i].w, hw);
        g[0] -= sol.x * hw[0] + sol.y * hw[1] + sol.z * hw[2];
        g[1] -= sol.x * hw[3] + sol.y * hw[4] + sol.z * hw[5];
        g[2] -= sol.x * hw[6] + sol.y * hw[7] + sol.z * hw[8];

//        g[3] -= sol.x * hw[9] + sol.y * hw[10] + sol.z * hw[11];
        g[3] = g[1];
        g[4] -= sol.x * hw[12] + sol.y * hw[13] + sol.z * hw[14];
        g[5] -= sol.x * hw[15] + sol.y * hw[16] + sol.z * hw[17];

//        g[6] -= sol.x * hw[18] + sol.y * hw[19] + sol.z * hw[20];
        g[6] = g[2];
//        g[7] -= sol.x * hw[21] + sol.y * hw[22] + sol.z * hw[23];
        g[7] = g[5];
        g[8] -= sol.x * hw[24] + sol.y * hw[25] + sol.z * hw[26]; 
    }
}

vec3 hrbfProjection(vec3 poslocal, vec3 normlocal, vec4 vertConf_neighbor[100], vec4 normRad_neighbor[100], int index_N)
{
    vec3 pose_proj = vec3(0.0,0.0,0.0);
    int max_iteration = 3;    

    //store all the neighbors in a array
    if(index_N == 0)
        return poslocal;

    vec3 update_pose = poslocal;

    for(int project_iteration = 0; project_iteration < max_iteration; project_iteration++)
    {
        vec3 point_sum = vec3(0.0,0.0,0.0);
        float weight = 0;
        float weight_sum = 0;
        for(int k = 0; k < index_N; k++)
        {

            float vx = update_pose.x - vertConf_neighbor[k].x;
            float vy = update_pose.y - vertConf_neighbor[k].y;
            float vz = update_pose.z - vertConf_neighbor[k].z;
            float dist2 = vx*vx + vy*vy + vz*vz;

            //weight = exp(-dist2/(normRad_neighbor[k].w * normRad_neighbor[k].w));
            weight = exp(-dist2/(0.0001));

            point_sum.x += vertConf_neighbor[k].x * weight;
            point_sum.y += vertConf_neighbor[k].y * weight;
            point_sum.z += vertConf_neighbor[k].z * weight;
            weight_sum += weight;
        }

        vec3 weighted_average = vec3(point_sum.x / weight_sum, point_sum.y / weight_sum, point_sum.z / weight_sum);
        vec3 grad = hrbfgradient(update_pose, vertConf_neighbor, normRad_neighbor, index_N);
        vec3 grad_n = normalize(grad);

        //vec3 grad_n = normlocal;
        float s = grad_n.x * (update_pose.x - weighted_average.x) + grad_n.y * (update_pose.y - weighted_average.y) + grad_n.z * (update_pose.z - weighted_average.z);
        pose_proj = vec3(update_pose.x - s * grad_n.x, update_pose.y - s * grad_n.y, update_pose.z - s * grad_n.z);

        update_pose = pose_proj;
    }
    //return pose_proj;
    return pose_proj;
}









