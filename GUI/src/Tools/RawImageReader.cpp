#include "RawImageReader.h"

RawImageLogReader::RawImageLogReader(std::string assofile, bool flipColors)
    :LogReader(assofile, flipColors)
{
    //Read association.txt file,
    std::cout << file << std::endl;
    assert(pangolin::FileExists(file.c_str()));
    std::ifstream fAssociation;
    fAssociation.open(file.c_str());

    while(!fAssociation.eof())
    {
       std::string s;
       getline(fAssociation,s);
       if(!s.empty())
       {
           std::stringstream ss;
           ss << s;
           double t;
           std::string sD, sRGB;
           ss >> t;
           vTimestamps.push_back(t);
           ss >> sD;
           vstrImageFilenamesD.push_back(sD);
           ss >> t;
           //vTimestamps.push_back(t);
           ss >> sRGB;
           vstrImageFilenamesRGB.push_back(sRGB);
       }
    }
    currentFrame = 0;
    numFrames = vstrImageFilenamesD.size();
}

RawImageLogReader::~RawImageLogReader()
{

}

void RawImageLogReader::getNext()
{

    getCore();
}

void RawImageLogReader::getBack()
{

}

int RawImageLogReader::getNumFrames()
{
    return numFrames;
}

bool RawImageLogReader::hasMore()
{
     return currentFrame < numFrames;
}

bool RawImageLogReader::rewound()
{

}

void RawImageLogReader::rewind()
{

}

void RawImageLogReader::fastForward(int frame)
{

}

const std::string RawImageLogReader::getFile()
{

}

void RawImageLogReader::setAuto(bool value)
{

}


void RawImageLogReader::getCore()
{
    //get image

    //get time stamp
    timestamp = int64_t(vTimestamps[currentFrame] * 1000000.0);
    currentFrame++;
}
