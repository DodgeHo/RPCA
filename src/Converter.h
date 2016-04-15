//
// Created by Marvin Hao on 2016-04-09.
//

#ifndef RPCA_CONVERTER_H
#define RPCA_CONVERTER_H

#include <opencv2/opencv.hpp>
#include <armadillo>
#include <boost/filesystem.hpp>
#include "RPCA.h"
#include <iomanip>
#include <sstream>
#include <string>


using namespace std;
using namespace cv;
using namespace arma;
using namespace boost::filesystem;

struct SCMat{
  /**
  * Single channel matrix
  */
  int rows;
  int cols;
  mat gray;
};

struct TCMat{
  /**
  * Three-channel matrix
  */
  int rows;
  int cols;
  mat red;
  mat green;
  mat blue;
};

struct SCVectorizedMat : public SCMat{
  /**
  * An SCMat which stores vectorized matrices.
  */
  int nMat = 0;
};

struct TCVectorizedMat : public TCMat{
  /**
  * An TCMat which stores vectorized matrices.
  */
  int nMat = 0;
};

struct TCVectorizedVidMat : public TCVectorizedMat{
  /**
  * An TCMat which stores vectorized frames from a video.
  */
  int ex;
  double fps;
  Size size;
};

class Converter {
public:
    static SCMat readGray2SCMat(string path);

    static SCMat gray2SCMat(cv::Mat& img);

    static void saveSCMat2Gray(string path, const SCMat& scmat);

    static cv::Mat SCMat2Gray(const SCMat& scmat);

    static SCVectorizedMat multiGray2SCMat(char* dirpath);

    static void saveSCMat2MultiGrey(char* resultDir, SCVectorizedMat& data);

    static TCMat readRGB2TCMat(string path);

    static TCMat RGB2TCMat(cv::Mat& img);

    static void saveTCMat2RGB(string path, const TCMat& tcmat);

    static cv::Mat TCMat2RGB(const TCMat& tcmat);

    static TCVectorizedMat multiRGB2TCMat(char* dirpath);

    static void saveTCMat2MultiRGB(char* resultDir, TCVectorizedMat& data);

    static TCVectorizedVidMat readRGBVideo2TCMat(char* path);

    static void saveTCMat2RGBVideo(char* resultDir, TCVectorizedVidMat& data);

    static void mergeImgToVid(char* src, char* dst, double fps);

private:

    static mat cvMat2AdMat(const cv::Mat& cvmat);

    static cv::Mat adMat2CvMat(const mat& admat);
};



#endif //RPCA_CONVERTER_H
