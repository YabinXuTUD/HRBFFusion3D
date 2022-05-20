#version 330 core

in vec2 texcoord;

out float FragColor;

uniform usampler2D gSampler;
uniform float cols;
uniform float rows;
uniform float depthFactor;
uniform float maxD;

//to make it compatible with the new input, maybe the depth scalar factor is 5000
//to adjust the parameter ,first we divide it to 1000 with float, so no precision is lost

float gaussD(float sigma, float x, float y)
{
	return exp(-((x*x+y*y)/(2.0f*sigma*sigma)));
}

void main()
{
	float depthFactor_adjustment = 1 / (depthFactor * 1000.0f);

    uint value_o = uint(texture(gSampler, texcoord.xy));

	float value = float(value_o) / depthFactor_adjustment;

    if(value > maxD * 1000.0f || value < 300.0f)
    {
        FragColor = 0.0;
    }
    else
    {
	    int x = int(texcoord.x * cols);
	    int y = int(texcoord.y * rows);

		const float sigma_space2_inv_half = 0.024691358;    // 0.5 / (sigma_space * sigma_space)
		const float sigma_color2_inv_half = 0.000555556;    // 0.5 / (sigma_color * sigma_color)

	    int R = 4;
	    int D = R * 2 + 1;

	    int tx = min(x - D / 2 + D, int(cols));
	    int ty = min(y - D / 2 + D, int(rows));

	    float sum1 = 0;
	    float sum2 = 0;

	    for(int cy = max(y - D / 2, 0); cy < ty; ++cy)
	    {
	        for(int cx = max(x - D / 2, 0); cx < tx; ++cx)
	        {
	            float texX = float(cx) / cols;
	            float texY = float(cy) / rows;

				//current depth
	            uint tmp_o = uint(texture(gSampler, vec2(texX, texY)));

				float tmp = float(tmp_o) / depthFactor_adjustment;

				if(tmp > 300.0 && abs(tmp - value) < 100)
				{
					float weight = gaussD(3.0, float(x) - float(cx), float(y) - float(cy));

					sum1 += float(tmp) * weight;
					sum2 += weight;
				}
				
	        }
	    }
	    FragColor = (sum1 / sum2) * depthFactor_adjustment;
    }
}
