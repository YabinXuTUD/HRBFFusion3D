/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */
 
#include "GPUTexture.h"

const std::string GPUTexture::RGB = "RGB";
const std::string GPUTexture::DEPTH_RAW = "DEPTH";
const std::string GPUTexture::DEPTH_FILTERED = "DEPTH_FILTERED";
const std::string GPUTexture::DEPTH_BILINEAR_FILTERED = "DEPTH_BILINEAR_FILTERED";
const std::string GPUTexture::DEPTH_METRIC = "DEPTH_METRIC";
const std::string GPUTexture::DEPTH_METRIC_FILTERED = "DEPTH_METRIC_FILTERED";
const std::string GPUTexture::DEPTH_NORM = "DEPTH_NORM";
const std::string GPUTexture::VERTEX_RAW = "VERTEX_RAW";
const std::string GPUTexture::VERTEX_FILTERED = "VERTEX_FILTERED";
const std::string GPUTexture::NORMAL = "NORMAL";
const std::string GPUTexture::NORMAL_OPT = "NORMAL_OPT";
const std::string GPUTexture::PRINCIPAL_CURV1 = "PRINCIPAL_CURV1";
const std::string GPUTexture::PRINCIPAL_CURV2 = "PRINCIPAL_CURV2";
const std::string GPUTexture::FILTERED_BY_MODEL_POINTS = "FILTERED_BY_MODEL_POINTS";
const std::string GPUTexture::GRADIENT_MAG = "GRADIENT_MAG";
const std::string GPUTexture::CONFIDENCE = "CONFIDENCE";
const std::string GPUTexture::RADIUS = "RADIUS";
const std::string GPUTexture::RADIUS_OPTIMIZED = "RADIUS_OPTIMIZED";

GPUTexture::GPUTexture(const int width,
                       const int height,
                       const GLenum internalFormat,
                       const GLenum format,
                       const GLenum dataType,
                       const bool draw,
                       const bool cuda)
 : texture(new pangolin::GlTexture(width, height, internalFormat, draw, 0, format, dataType)),
   draw(draw),
   width(width),
   height(height),
   internalFormat(internalFormat),
   format(format),
   dataType(dataType)
{
    if(cuda)
    {
        cudaGraphicsGLRegisterImage(&cudaRes, texture->tid, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
    }
    else
    {
        cudaRes = 0;
    }
}

GPUTexture::~GPUTexture()
{
    if(texture)
    {
        delete texture;
    }

    if(cudaRes)
    {
        cudaGraphicsUnregisterResource(cudaRes);
    }
}
