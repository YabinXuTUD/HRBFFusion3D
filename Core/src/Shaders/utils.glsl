
float compute_dist(vec3 p1, vec3 p2)
{
    float vx = p1.x - p2.x;
    float vy = p1.y - p2.y;
    float vz = p1.z - p2.z;
    float dist = sqrt(vx*vx + vy*vy + vz*vz);
    return dist;
}

float getlength(vec3 p)
{
     float len = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
     return len;
}

float angleBetween(vec3 a, vec3 b)
{
    return acos(dot(a, b) / (length(a) * length(b)));
}

bool checkNeighbours(vec2 texCoord, sampler2D depth) 
{
    float z = float(textureLod(depth, vec2(texCoord.x - (1.0 / cols), texCoord.y), 0.0));
    if(z == 0)
        return false;
        
    z = float(textureLod(depth, vec2(texCoord.x, texCoord.y - (1.0 / rows)), 0.0));
    if(z == 0)
        return false;

    z = float(textureLod(depth, vec2(texCoord.x + (1.0 / cols), texCoord.y), 0.0));
    if(z == 0)
        return false;
        
    z = float(textureLod(depth, vec2(texCoord.x, texCoord.y + (1.0 / rows)), 0.0));
    if(z == 0)
        return false;
        
    return true;
}

bool check_ifsupport(vec3 p, vec4 vertConf_neighbor[100], vec4 normRad_neighbor[100], int index_N, int min_support_num)
{
     bool support = false;
     int count = 0;
     for(int i = 0 ; i < index_N; i++)
     {
         if(getlength(p - vertConf_neighbor[i].xyz) < normRad_neighbor[i].w)
         {
            support = true;
            count ++;
         }
     }
     if(count > min_support_num)
        return true;
     else
        return false;
}


void bubblesort(inout float array[100], inout int indices[100], int maxlen, int sort_k_largest)
{
	float temp_value;
	int temp_index;

	for(int i = 0;i < sort_k_largest; i++) 
	{
		for(int j = i + 1;j < maxlen; j++) 
		{
			if(array[i] < array[j])
			{
				temp_value = array[i];
				array[i] = array[j];
				array[j] = temp_value;

				temp_index = indices[i];
				indices[i] = indices[j];
				indices[j] = temp_index;
			}
		}
	}
}


// void swap(inout float a, inout float b, inout int index_1, inout int index_2)
// {
// 	float temp;

// 	temp = a;
// 	a = b;
// 	b = temp;

// 	int temp_index;

// 	temp_index = index_1;
// 	index_1 = index_2;
// 	index_2 = temp_index;

// 	return;
// }



// void quicksort(inout float array[100], inout int indeces[100], int maxlen, int begin, int end)
// {
// 	int i, j;

// 	if (begin < end)
// 	{
// 		i = begin + 1;  
// 		j = end;     

// 		while (i < j)
// 		{
// 			if (array[i] > array[begin])  
// 			{
// 				swap(array[i], array[j], indeces[i], indeces[j]);  
// 				j--;
// 			}
// 			else
// 			{
// 				i++;  
// 			}
// 		}
// 		if (array[i] >= array[begin])  
// 		{
// 			i--;
// 		}

// 		swap(array[begin], array[i], indeces[begin], indeces[i]);    

// 		quicksort(array, indeces, maxlen, begin, i);
// 		quicksort(array, indeces, maxlen, j, end);
// 	}
// }