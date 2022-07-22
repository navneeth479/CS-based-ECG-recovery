// bi-linear texture sampler
__constant sampler_t linearSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__constant float2 GRID_SPACING = {1.0f, 1.0f};


__kernel void backproject(
             read_only image2d_t sinogram,
             write_only image2d_t reco,
             float deltaS,
             float deltaTheta,
             float sino_o_x,
             float sino_o_y,
             float projections,
             float detector_len,
             float spacing_x,
             float spacing_y,
             float o_x,
             float o_y
             )
             {
                float maxTheta = get_image_width(sinogram);
                float maxS = get_image_height(sinogram);

                int x = get_global_id(0); // out buffer size
                int y = get_global_id(1); // out buffer size
                int gridSizeX = get_global_size(0);
                int gridSizeY = get_global_size(1);

                if (x >= gridSizeX || y >= gridSizeY)
                     return;

                // Convert pixel to physical coords
                float xcoord =  (x * spacing_x) + o_x;
                float ycoord =  (y * spacing_y) + o_y;

                for (float i = 0.0; i < maxTheta; i+=1.0f)
                {
                  float theta = (deltaTheta * i);
                  float cosTheta = cos(theta);
                  float sinTheta = sin(theta);
                  float s = xcoord * cosTheta + ycoord * sinTheta; // physical position

                  // Convert detector physical position back to pixel coord
                  float s_p = (s - sino_o_y)/deltaS;

                  float proj_val = read_imagef(sinogram, linearSampler, (float2)(i + 0.5f, s_p + 0.5f)).x;
                  write_imagef(reco, (int2)(x,y), proj_val);
                }
             }