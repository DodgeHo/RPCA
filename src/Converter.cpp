//
// Created by Marvin Hao on 2016-04-09.
//

#include "Converter.h"
#include <chrono>

SCMat Converter::readGray2SCMat(string path){
    /**
    * Read image and convert it into a single-channel matrix.
    */
    cv::Mat inImg = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    return gray2SCMat(inImg);
}

SCMat Converter::gray2SCMat(cv::Mat& img){
    /**
    * Convert an img (which is in cv::Mat format) into a single-channel matrix.
    */
    SCMat scmat;
    scmat.gray = cvMat2AdMat(img);
    scmat.rows = scmat.gray.n_rows;
    scmat.cols = scmat.gray.n_cols;
    return scmat;
}

void Converter::saveSCMat2Gray(string path, const SCMat& scmat){
    /**
    * Save an img whose data is stored in an SCMat.
    */
    imwrite(path, SCMat2Gray(scmat));
}

cv::Mat Converter::SCMat2Gray(const SCMat& scmat){
    /**
    * Convert an SCMat to an OpenCV image.
    */
    return adMat2CvMat(scmat.gray);
}

SCVectorizedMat Converter::multiGray2SCMat(char* dirpath){
    /**
    * Vectorize and concatenate multiple images.
    */
    SCVectorizedMat data;
    path p(dirpath);

    if (exists(p) && is_directory(p)){
        for (directory_entry& x : directory_iterator(p)){
            if (is_regular_file(x.path())){
                string path = x.path().string();
                if (path.find( ".DS_Store" ) != string::npos )
                    continue;
                SCMat frame = readGray2SCMat(path);
                cout << path << endl;
                frame.gray.reshape(frame.rows * frame.cols, 1);
                if (data.nMat == 0) {
                    data.rows = frame.rows;
                    data.cols = frame.cols;
                    data.gray = frame.gray;
                    data.nMat ++;
                }
                else{
                    data.gray = join_rows(data.gray, frame.gray);
                    data.nMat ++;
                }
            }
        }
    }
    return data;
}

void Converter::saveSCMat2MultiGrey(char* resultDir, SCVectorizedMat& data){
    path p(resultDir);
    if (exists(p) && is_directory(p)){
        string pathBase = p.string();

        for (size_t i = 0 ; i < data.nMat ; ++i){
            // Generate self-incremented file names.
            ostringstream ss;
            ss << std::setw( 8 ) << std::setfill( '0' ) << i;
            string fullPath = pathBase + "/" + ss.str() + ".jpg";

            // Devectorization
            SCMat frame;
            frame.rows = data.rows;
            frame.cols = data.cols;
            frame.gray = data.gray.col(i);
            frame.gray.reshape(data.rows, data.cols);
            saveSCMat2Gray(fullPath, frame);
        }
    }
}

TCMat Converter::readRGB2TCMat(string path){
    /**
    * Read an RGB image and convert it into a single-channel matrix.
    */
    cv::Mat inImg = imread(path, CV_LOAD_IMAGE_COLOR);
    return RGB2TCMat(inImg);
}

TCMat Converter::RGB2TCMat(cv::Mat& img){
    /**
    * Convert an RGB img (which is in cv::Mat format) into a three-channel matrix.
    */
    cv::Mat channels[3];
    cv::split(img, channels);

    // Note that OpenCV deploys BGR color arrangement.
    TCMat tcmat;
    tcmat.blue = cvMat2AdMat(channels[0]);
    tcmat.green = cvMat2AdMat(channels[1]);
    tcmat.red = cvMat2AdMat(channels[2]);
    tcmat.rows = tcmat.blue.n_rows;
    tcmat.cols = tcmat.blue.n_cols;
    return tcmat;
}

void Converter::saveTCMat2RGB(string path, const TCMat& tcmat){
    /**
    * Save an img whose data is stored in an TCMat.
    */
    imwrite(path, TCMat2RGB(tcmat));
}

cv::Mat Converter::TCMat2RGB(const TCMat& tcmat){
    /**
    * Convert an TCMat to an OpenCV image.
    */
    cv::Mat img;
    cv::Mat channels[3];

    // Note that OpenCV deploys BGR color arrangement.
    channels[0] = adMat2CvMat(tcmat.blue);
    channels[1] = adMat2CvMat(tcmat.green);
    channels[2] = adMat2CvMat(tcmat.red);

    cv::merge(channels, 3, img);
    return img;
}

mat Converter::cvMat2AdMat(const cv::Mat& cvmat){
    /**
     * Convert an OpenCV Mat to an Armadillo mat.
     * The OpenCV Mat entries should be uchars.
     * Note that Armadillo's mat is Mat<double>
     */

    // Set to column-based order.
    cv::Mat cvmatTmp = cvmat.t();
    arma::Mat<uchar> admatUchar(reinterpret_cast<uchar*> (cvmatTmp.data), cvmat.rows, cvmat.cols);
    return conv_to<mat>::from(admatUchar);
}

cv::Mat Converter::adMat2CvMat(const mat& admat){
    /**
     * Convert and Armadillo mat to OpenCV Mat.
     * The converted CV Mat will be filled with uchar elements.
     */
    Mat<uchar> ucharAdMat_t = conv_to<Mat<uchar>>::from(admat.t());

    // Without deep copy, The cv::Mat will share the data memory with arma::mat, which will be deconstructed beyond this method's scope.
    return cv::Mat(admat.n_rows, admat.n_cols, CV_8U, ucharAdMat_t.memptr()).clone();
}

TCVectorizedMat Converter::multiRGB2TCMat(char* dirpath){
  /**
  * Vectorize and concatenate multiple images.
  */
  TCVectorizedMat data;
  path p(dirpath);
  std::vector<TCMat> rawData;
  bool firstImg = true;

  if (exists(p) && is_directory(p)){
      for (directory_entry& x : directory_iterator(p)){
          if (is_regular_file(x.path())){
              string path = x.path().string();
              if (path.find( ".DS_Store" ) != string::npos )
                  continue;
              TCMat frame = readRGB2TCMat(path);
              if (firstImg){
                  data.rows = frame.rows;
                  data.cols = frame.cols;
                  firstImg = false;
              }
              frame.red.reshape(frame.rows * frame.cols, 1);
              frame.green.reshape(frame.rows * frame.cols, 1);
              frame.blue.reshape(frame.rows * frame.cols, 1);
              rawData.push_back(frame);
              data.nMat ++;
          }
      }
  }


  data.red = mat(data.rows * data.cols, data.nMat);
  data.blue = mat(data.rows * data.cols, data.nMat);
  data.green = mat(data.rows * data.cols, data.nMat);

  for(int i = 0; i < data.nMat; ++i){
    data.red.col(i) = rawData[i].red;
    data.blue.col(i) = rawData[i].blue;
    data.green.col(i) = rawData[i].green;
  }
  return data;
}

void Converter::saveTCMat2MultiRGB(char* resultDir, TCVectorizedMat& data){
    path p(resultDir);
    if (exists(p) && is_directory(p)){
        string pathBase = p.string();

        for (size_t i = 0 ; i < data.nMat ; ++i){
            // Generate self-incremented file names.
            ostringstream ss;
            ss << std::setw( 8 ) << std::setfill( '0' ) << i;
            string fullPath = pathBase + "/" + ss.str() + ".jpg";

            // Devectorization
            TCMat frame;
            frame.rows = data.rows;
            frame.cols = data.cols;
            frame.green = data.green.col(i);
            frame.green.reshape(data.rows, data.cols);
            frame.red = data.red.col(i);
            frame.red.reshape(data.rows, data.cols);
            frame.blue = data.blue.col(i);
            frame.blue.reshape(data.rows, data.cols);
            saveTCMat2RGB(fullPath, frame);
        }
    }
}

TCVectorizedVidMat Converter::readRGBVideo2TCMat(char* path){
    VideoCapture cap(path);
    if (!cap.isOpened()){
        cout << "Cannot read the video." << endl;
        throw 1;
    }

    TCVectorizedVidMat data;
    data.ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
    data.fps = cap.get(CV_CAP_PROP_FPS);
    data.size = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    data.rows = (int)data.size.height;
    data.cols = (int)data.size.width;

    cv::Mat cvFrame;
    std::vector<TCMat> rawData;

    while (1){
      // auto start = std::chrono::high_resolution_clock::now();

        cap >> cvFrame;
        if (cvFrame.empty())
          break;
        TCMat frame = RGB2TCMat(cvFrame);

        frame.red.reshape(frame.rows * frame.cols, 1);
        frame.green.reshape(frame.rows * frame.cols, 1);
        frame.blue.reshape(frame.rows * frame.cols, 1);
        rawData.push_back(frame);

        data.nMat ++;
        // auto finish = std::chrono::high_resolution_clock::now();
        // auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(finish-start);
        // cout << microseconds.count() << endl;
    }



    data.red = mat(data.rows * data.cols, data.nMat);
    data.blue = mat(data.rows * data.cols, data.nMat);
    data.green = mat(data.rows * data.cols, data.nMat);

    for(int i = 0; i < data.nMat; ++i){
      data.red.col(i) = rawData[i].red;
      data.blue.col(i) = rawData[i].blue;
      data.green.col(i) = rawData[i].green;
    }

    return data;
}

void Converter::saveTCMat2RGBVideo(char* resultDir, TCVectorizedVidMat& data){
    VideoWriter outputVideo;
    outputVideo.open(resultDir, data.ex, data.fps, data.size, true);
    if (!outputVideo.isOpened()){
        cout << "cannot save video." << endl;
        throw 1;
    }

    cout << "start save"<<endl;

    for (int i = 0 ; i < data.nMat ; ++i){
        // Devectorization
        cout << i << endl;
        TCMat frame;
        frame.rows = data.rows;
        frame.cols = data.cols;
        frame.green = data.green.col(i);
        frame.green.reshape(data.rows, data.cols);
        frame.red = data.red.col(i);
        frame.red.reshape(data.rows, data.cols);
        frame.blue = data.blue.col(i);
        frame.blue.reshape(data.rows, data.cols);
        cv::Mat img = TCMat2RGB(frame);
        imshow("Video", img);
        int key = waitKey(10);
           if((char)key == 'q') { break; }
        outputVideo << img;
    }
    return;
}

void Converter::mergeImgToVid(char* src, char* dst, double fps){
  path p(src);
  VideoWriter output;
  bool opened = false;

  if (exists(p) && is_directory(p)){
      for (directory_entry& x : directory_iterator(p)){
          if (is_regular_file(x.path())){
              string path = x.path().string();
              if (path.find( ".DS_Store" ) != string::npos )
                  continue;
              cv::Mat inImg = imread(path, CV_LOAD_IMAGE_COLOR);
              if (!output.isOpened()){
                output.open(dst, CV_FOURCC('M', 'J', 'P', 'G'), fps, inImg.size(), true);
              }
              output.write(inImg);
          }
      }
  }
}
