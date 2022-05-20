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
 
#ifndef UNIFORM_H_
#define UNIFORM_H_

#include <string>
#include <algorithm>
#include <iostream>
#include <Eigen/Core>

//What is uniform a way for us to get data from CPU side of things in c++ ways into our shader use like a viriable

class Uniform
{
    public:
        Uniform(const std::string & id, const int & v)
         : id(id),
           i(v),
           t(INT)
        {}

        Uniform(const std::string & id, const float & v)
         : id(id),
           f(v),
           t(FLOAT)
        {}

        Uniform(const std::string & id, const Eigen::Vector2f & v)
         : id(id),
           v2(v),
           t(VEC2)
        {}

        Uniform(const std::string & id, const Eigen::Vector3f & v)
         : id(id),
           v3(v),
           t(VEC3)
        {}

        Uniform(const std::string & id, const Eigen::Vector4f & v)
         : id(id),
           v4(v),
           t(VEC4)
        {}

        Uniform(const std::string & id, const Eigen::Matrix4f & v)
         : id(id),
           m4(v),
           t(MAT4)
        {}

        Uniform(const std::string & id, std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f>> v)
         : id(id),
           t(ARRAY_MAT4)
        {
            if(v.size() > 200)
                std::cout << "add additional array\n";
            m4array = new float[3200];
            int N_KF = v.size();
            int count = 0;

            Eigen::Matrix4f Iden = Eigen::Matrix4f::Identity();
            for (int i = 0; i < N_KF; i++) {
                for (int j = 0; j < 4; j++) {
                    for (int k = 0; k < 4; k++) {
                        m4array[count] = v[i](k, j);
                        count++;
                    }
                }
            }
        }

        std::string id;

        int i;
        float f;
        Eigen::Vector2f v2;
        Eigen::Vector3f v3;
        Eigen::Vector4f v4;
        Eigen::Matrix4f m4;

        float* m4array;
        enum Type
        {
            INT,
            FLOAT,
            VEC2,
            VEC3,
            VEC4,
            MAT4,
            ARRAY_MAT4,
            NONE
        };

        Type t;
};


#endif /* UNIFORM_H_ */
