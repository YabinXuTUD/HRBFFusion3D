#ifndef RAWIMAGELOGREADER_H
#define RAWIMAGELOGREADER_H

#include <Utils/Resolution.h>
#include <Utils/Stopwatch.h>
#include <pangolin/utils/file_utils.h>

#include "LogReader.h"

#include <string>
#include <fstream>

class RawImageLogReader: public LogReader
{
public:
    RawImageLogReader(std::string assofile, bool flipColors);

    virtual ~RawImageLogReader();

    void getNext();

    void getBack();

    int getNumFrames();

    bool hasMore();

    bool rewound();

    void rewind();

    void fastForward(int frame);

    const std::string getFile();

    void setAuto(bool value);

    std::vector<std::string> vstrImageFilenamesRGB;
    std::vector<std::string> vstrImageFilenamesD;
    std::vector<double> vTimestamps;

private:
    void getCore();

};

#endif // RAWLOGREADERUNCOMPRESSED_H
