#include "TrajectoryManager.h"



TrajectoryManager::TrajectoryManager(){


};
TrajectoryManager::~TrajectoryManager(){

};

Eigen::Vector3f TrajectoryManager::rodrigues2(const Eigen::Matrix3f& matrix)
{
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

    double rx = R(2, 1) - R(1, 2);
    double ry = R(0, 2) - R(2, 0);
    double rz = R(1, 0) - R(0, 1);

    double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
    double c = (R.trace() - 1) * 0.5;
    c = c > 1. ? 1. : c < -1. ? -1. : c;

    double theta = acos(c);

    //not easy to understand if rotation matrix is a symmetric matrix?!!!
    if( s < 1e-5 )
    {
        double t;

        if( c > 0 )
            rx = ry = rz = 0;
        else
        {
            t = (R(0, 0) + 1)*0.5;
            rx = sqrt( std::max(t, 0.0) );
            t = (R(1, 1) + 1)*0.5;
            ry = sqrt( std::max(t, 0.0) ) * (R(0, 1) < 0 ? -1.0 : 1.0);
            t = (R(2, 2) + 1)*0.5;
            rz = sqrt( std::max(t, 0.0) ) * (R(0, 2) < 0 ? -1.0 : 1.0);

            if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0) )
                rz = -rz;
            theta /= sqrt(rx*rx + ry*ry + rz*rz);
            rx *= theta;
            ry *= theta;
            rz *= theta;
        }
    }
    else
    {
        double vth = 1/(2*s);
        vth *= theta;
        rx *= vth; ry *= vth; rz *= vth;
    }
    return Eigen::Vector3d(rx, ry, rz).cast<float>();
}

void TrajectoryManager::LoadFromFile(){
    FILE* fp;
    std::string trajFile = GlobalStateParam::get().globalInputTrajectoryFile;
    std::cout << "open file: " << trajFile <<"\n";
    fp = fopen(trajFile.c_str(), "r");

    if(fp == nullptr){
        std::cout << "can not open trajactory file, do not use groudtruth trajactory" << std::endl;
        return;
    }
    if(GlobalStateParam::get().globalInputTrajectoryFormat == "zhou")
    {
        std::cout << "--------------------Load QianYi Zhou trajectory format------------------------ " << std::endl;
        while(!feof(fp))
        {
            float a_00, a_10, a_20, a_30, a_01, a_11, a_21, a_31, a_02, a_12, a_22, a_32, a_03, a_13, a_23, a_33;
            int i_1, i_2, i_3;
            Eigen::Matrix4f ground_pose;
            fscanf(fp , "%d\t%d\t%d", &i_1, &i_2, &i_3);
            fscanf(fp, " %f %f %f %f", &a_00, &a_01, &a_02, &a_03);
            fscanf(fp, " %f %f %f %f", &a_10, &a_11, &a_12, &a_13);
            fscanf(fp, " %f %f %f %f", &a_20, &a_21, &a_22, &a_23);
            fscanf(fp, " %f %f %f %f\n", &a_30, &a_31, &a_32, &a_33);
            ground_pose << a_00, a_01, a_02, a_03, a_10, a_11, a_12, a_13, a_20, a_21, a_22, a_23, a_30, a_31, a_32, a_33;
            poses.push_back(ground_pose);
        }
        //fclose(fp);
        //because the first pose in the trajectory file is not identity matrix
        for(int i = 1 ; i < poses.size(); i++)
        {
            poses[i] = poses[0].inverse() * poses[i];
        }
        poses[0] = Eigen::Matrix4f::Identity();
    }

    if(GlobalStateParam::get().globalInputTrajectoryFormat == "ICL_NUIM_RT")
    {
        std::cout << "--------------------ICL_RT------------------------ " << std::endl;
        Eigen::Matrix4f trans;
        trans << 1.0, 0.0, 0.0, 0.0,
                 0.0, -1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, 1.0;
        Eigen::Matrix4f trans1;
        trans1 << -1.0, 0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, 1.0;

        while(!feof(fp))
        {
            float a_00, a_10, a_20, a_30, a_01, a_11, a_21, a_31, a_02, a_12, a_22, a_32, a_03, a_13, a_23, a_33;
            int i_1, i_2, i_3;
            Eigen::Matrix4f ground_pose;
            fscanf(fp, "%f %f %f %f", &a_00, &a_01, &a_02, &a_03);
            fscanf(fp, "%f %f %f %f", &a_10, &a_11, &a_12, &a_13);
            fscanf(fp, "%f %f %f %f\n", &a_20, &a_21, &a_22, &a_23);
            ground_pose << a_00, a_01, a_02, a_03, a_10, a_11, a_12, a_13, a_20, a_21, a_22, a_23, 0.0f, 0.0f, 0.0f, 1.0f;
            poses.push_back(ground_pose);
        }
        for(int i = 0; i < poses.size(); i++)
        {
            poses[i] = trans1 * poses[i] * trans;
        }
    }

    if(GlobalStateParam::get().globalInputTrajectoryFormat == "lefloch")
    {
        std::cout << "--------------------Load TPAMI2017 trajectory format------------------------ " << std::endl;
        Eigen::Matrix4f T_1, T_2, pose_init;

        //pose_init << 1,0,0,0,0,-1,0,0,0,0,-1,0,0,0,0,1;

        pose_init << 1.000000, -0.000228, 0.000007, 0.023992,-0.000228, -0.999970, 0.007753, 0.003886, 0.000005, -0.007753, -0.999970, -0.001633, 0.000000, 0.000000,0.000000, 1.000000;
        while(!feof(fp))
        {
            //pose file from T
            int index;
            float a_00, a_10, a_20, a_30, a_01, a_11, a_21, a_31, a_02, a_12, a_22, a_32, a_03, a_13, a_23, a_33;
            Eigen::Matrix4f ground_pose;
            fscanf(fp, "%d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",&index,
                   &a_00, &a_10, &a_20, &a_30,
                   &a_01, &a_11, &a_21, &a_31,
                   &a_02, &a_12, &a_22, &a_32,
                   &a_03, &a_13, &a_23, &a_33);
            ground_pose << a_00, a_01, a_02, a_03, a_10, a_11, a_12, a_13, a_20, a_21, a_22, a_23, a_30, a_31, a_32, a_33;
            //ground_pose << a_00, a_10, a_20, a_30, a_01, a_11, a_21, a_31, a_02, a_12, a_22, a_32, a_03, a_13, a_23, a_33;
            ground_pose = /*pose_init.inverse() **/ ground_pose * pose_init; //compatible to our current corrdinate ,maybe with different coordinate frames
//            /Eigen::Matrix4f ground_pose_inv = ground_pose.inverse();
            poses.push_back(ground_pose);
        }
    }
    fclose(fp);

    if(GlobalStateParam::get().globalInputTrajectoryFormat == "CoRBS")
    {
        std::cout << "--------------------Load CoRBS trajectory format------------------------" << std::endl;
        std::ifstream file;
        std::string line;
        file.open(trajFile.c_str());

        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        int count = 0;
        while(!file.eof())
        {
           unsigned long long int utime;                //timestamp
           float x, y, z, qx, qy, qz, qw;

           std::getline(file, line);
           if(line.empty() || line[0] == '#') continue;
           size_t first_spacePos = line.find_first_of(" ");
           std::remove(line.begin(), line.begin() + first_spacePos, '.');
           int n = sscanf(line.c_str(), "%llu %f %f %f %f %f %f %f", &utime, &x, &y, &z, &qx, &qy, &qz, &qw);
           if(file.eof())
               break;
           assert(n==8);
           Eigen::Quaternionf q(qw, qx, qy, qz);        //set quaternion to Eigen for transformation
           Eigen::Vector3f t(x, y, z);
           Eigen::Isometry3f T;
           T.setIdentity();
           T.pretranslate(t).rotate(q);
           pose = T.matrix();

           poses.push_back(pose);
           timstamp.push_back(utime);
           count++;
        }
        file.close();
        poses_original = poses;
        for (int i = 0; i < 10; i++){
            std::cout << poses[i] << std::endl;
        }
    }

    if(GlobalStateParam::get().globalInputTrajectoryFormat == "TUM")
    {
        std::cout << "--------------------Load TUM trajectory format------------------------ " << std::endl;     
        std::ifstream file;
        std::string line;
        file.open(trajFile.c_str());

        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        int count = 0;
        Eigen::Matrix4f pose_init = Eigen::Matrix4f::Identity();
        while(!file.eof())
        {
           unsigned long long int utime;                //timestamp
           float x, y, z, qx, qy, qz, qw;

           std::getline(file, line);
           if(line.empty() || line[0] == '#') continue;
           size_t first_spacePos = line.find_first_of(" ");
           std::remove(line.begin(), line.begin() + first_spacePos, '.');
           int n = sscanf(line.c_str(), "%llu %f %f %f %f %f %f %f", &utime, &x, &y, &z, &qx, &qy, &qz, &qw);

           if(file.eof())
               break;
           assert(n==8);

           Eigen::Quaternionf q(qw, qx, qy, qz);        //set quaternion to Eigen for transformation
           Eigen::Vector3f t(x, y, z);

           Eigen::Isometry3f T;
           T.setIdentity();
           T.pretranslate(t).rotate(q);

           bool TUM_Dataset = false;
           if(TUM_Dataset)
           {
               // Poses are stored in the file in iSAM basis, undo it
               Eigen::Matrix4f pose_capture_sys = T.matrix();

               Eigen::Matrix4f M;
               M <<  0,  0, 1, 0,
                    -1,  0, 0, 0,
                     0, -1, 0, 0,
                     0,  0, 0, 1;
               pose = M.inverse() * T * M;
               pose = T.matrix();
           }else {
               pose = T.matrix();
           }

           poses.push_back(pose);
           timstamp.push_back(utime);
           count++;
        }
        file.close();
        poses_original = poses;
        for (int i = 0; i < 10; i++){
            std::cout << poses[i] << std::endl;
        }
    }
    std::cout << "--------------------The trajectory file is specified------------------------ " << std::endl;

    if(GlobalStateParam::get().globalInputTrajectoryFormat == "open3d")
    {
        std::cout << "-------------load trajectory from open3d pose graph file------------" << std::endl;
//        open3d::ReadPoseGraph(trajFile, pose_graph_);

        //transfer to poses;

//        for(int i = 0; i < pose_graph_.nodes_.size(); i++)
//        {
//            poses.push_back(pose_graph_.nodes_[i].pose_.cast<float>());
//        }
//        std::cout << "load " << pose_graph_.nodes_.size() << " nodes and " << pose_graph_.edges_.size() << " edges" << std::endl;
    }
    //compute trajectory length, average velocity and rotation angle
    float sum_trans = 0;
    float sum_rot = 0;
    for(int i = 0; i < poses.size() - 1; i++)
    {
        Eigen::Vector3f DiffTrans = (poses[i + 1].inverse() * poses[i]).topRightCorner(3, 1);
        sum_trans = sum_trans + DiffTrans.norm();
        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> DiffRot = (poses[i + 1].inverse() * poses[i]).topLeftCorner(3, 3);
        sum_rot = sum_rot + rodrigues2(DiffRot).norm();
    }
    std::cout << "total trajectory length: " << sum_trans  << std::endl;
    std::cout << "average velocity: " << sum_trans / (poses.size() / 30.0) << std::endl;
    std::cout << "average rotation: " << sum_rot * 180 / (3.1415926 * poses.size() / 30.0) << std::endl;
}

bool TrajectoryManager::SaveTrajectoryToFile(){
    std::cout << "--------------------save trajectory----------------" << std::endl;
    std::string fname;
    if(GlobalStateParam::get().globalOutputSaveTrjectoryFileType == "zhou")
    {
        fname = "hrbf_trajectory.txt";
        FILE * f = fopen(fname.c_str(), "w");
        if (f == nullptr) {
            std::cout << "Write LOG failed: unable to open file: " << fname.c_str() << std::endl;
            return false;
        }
        //if we should use double or float???
        for(size_t i = 0; i < poses.size(); i++)
        {
            Eigen::Matrix4f trans = poses[i];
            fprintf(f, "%d %d %d\n", static_cast<int>(i), static_cast<int>(i), static_cast<int>(i + 1));
            fprintf(f, "%f %f %f %f\n", trans(0,0), trans(0,1), trans(0,2),
                    trans(0,3));
            fprintf(f, "%f %f %f %f\n", trans(1,0), trans(1,1), trans(1,2),
                    trans(1,3));
            fprintf(f, "%f %f %f %f\n", trans(2,0), trans(2,1), trans(2,2),
                    trans(2,3));
            fprintf(f, "%f %f %f %f\n", trans(3,0), trans(3,1), trans(3,2),
                    trans(3,3));
        }
        fclose(f);
        std::cout << "--------------------trajectory saved; format: Zhou----------------" << std::endl;
        return true;
    }
    if(GlobalStateParam::get().globalOutputSaveTrjectoryFileType == "TUM")
    {
        fname = "hrbf_trajectory.freiburg";
        std::ofstream f;
        f.open(fname.c_str(), std::fstream::out);

        std::cout << "pose size: "<< poses.size() << std::endl;
        for (size_t i = 0;i < poses.size(); i++) {
            Eigen::Vector3f trans = poses[i].topRightCorner(3, 1);
            Eigen::Matrix3f rot = poses[i].topLeftCorner(3, 3);

            std::stringstream strs;
            if(GlobalStateParam::get().globalInputICLNUIMDataset)     //export timestamp o for ICL dataset 0 for TUM
            {
               //strs << std::setprecision(6) << std::fixed << timstamp[i] << " ";
               strs << int(timstamp[i]) << " ";
               trans(1) = - trans(1);
            }
            else
            {
//                std::cout << timstamp[i] << "\n";
                strs << std::setprecision(6) << std::fixed << (double)timstamp[i] / 1000000.0 << " ";
            }

            f << strs.str() << trans(0) << " " << trans(1) << " " << trans(2) << " ";

            Eigen::Quaternionf currentCameraRotation(rot);
            f << currentCameraRotation.x() << " " << currentCameraRotation.y() << " " << currentCameraRotation.z() << " " << currentCameraRotation.w() << "\n";
        }
        f.close();
        std::cout << "--------------------trajectory saved; format: TUM----------------" << std::endl;
        return true;
    }

    if(GlobalStateParam::get().globalOutputSaveTrjectoryFileType == "lefloch")
    {
        fname = "hrbf_trajectory_lef.txt";
        std::ofstream f;
        f.open(fname.c_str(), std::fstream::out);
        for (size_t i = 0;i < poses.size(); i++)
        {
            Eigen::Vector3f trans = poses[i].topRightCorner(3, 1);
            Eigen::Matrix3f rot = poses[i].topLeftCorner(3, 3);

            f << i << " ";
            for(int j = 0; j < 4; j++)
            {
                for(int k = 0; k < 4; k++)
                {
                    f << poses[i](k, j) << " ";
                }
            }
            f << "\n";
        }
        f.close();
    }
    return false;
}



