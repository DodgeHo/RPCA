#include "src/RPCA.h"
#include "src/Converter.h"
using namespace std;
using namespace cv;
using namespace arma;

int main(int argc, char **argv) {

    //  TCMat data = Converter::readRGB2TCMat("/Users/Marvin/Documents/mycode/cpp/rpca/sky_contaminated.jpg");
    //  InexactRPCASolver r(data.red);
    //  r.solve();
    //
    //  InexactRPCASolver g(data.green);
    //  g.solve();
    //
    //  InexactRPCASolver b(data.blue);
    //  b.solve();
    //
    //  TCMat lowrank = data;
    //  lowrank.red = r.getLowRank();
    //  lowrank.blue = b.getLowRank();
    //  lowrank.green = g.getLowRank();
    //
    //
    //  Converter::saveTCMat2RGB("/Users/Marvin/Documents/mycode/cpp/rpca/result.jpg", lowrank);

    //
    TCVectorizedMat data = Converter::multiRGB2TCMat("/Users/Marvin/Documents/mycode/cpp/rpca/imgs");
    cout << "copy done" << endl;

    RPInexactRPCASolver r(data.red, 0.01, 3);
    r.solve();

    RPInexactRPCASolver g(data.green, 0.01, 3);
    g.solve();

    RPInexactRPCASolver b(data.blue, 0.01, 3);
    b.solve();

    TCVectorizedMat lowrank, sparse;
    lowrank.rows = data.rows;
    lowrank.cols = data.cols;
    lowrank.nMat = data.nMat;
    lowrank.red = r.getLowRank();
    lowrank.green = g.getLowRank();
    lowrank.blue = b.getLowRank();

    sparse.rows = data.rows;
    sparse.cols = data.cols;
    sparse.nMat = data.nMat;
    sparse.red = r.getSparse();
    sparse.green = g.getSparse();
    sparse.blue = b.getSparse();

    Converter::saveTCMat2MultiRGB("/Users/Marvin/Documents/mycode/cpp/rpca/result/lowrank", lowrank);
    Converter::saveTCMat2MultiRGB("/Users/Marvin/Documents/mycode/cpp/rpca/result/sparse", sparse);

    // mat admat(2, 3);
    // admat << 1 << 2 << 3 << endr
    // << 4 << 5 << 6 <<endr;
    //
    // cv::Mat cvmat(Converter::adMat2CvMat(admat));
    //
    // mat finalMat(Converter::cvMat2AdMat(cvmat));
    // cout << cvmat;

    // cv::Mat A = cv::Mat::eye(2, 3, CV_8U);
    // mat B = Converter::cvMat2AdMat(A);
    // cv::Mat C = Converter::adMat2CvMat(B);
    // cout << A <<endl << C <<endl;

    // TCVectorizedVidMat data = Converter::readRGBVideo2TCMat("/Users/Marvin/Documents/mycode/cpp/rpca/video/Meet_Crowd.mpg");
    // cout << "copy done" << endl;
    //
    // InexactRPCASolver r(data.red, 0.05);
    // r.solve();
    //
    // InexactRPCASolver g(data.green, 0.05);
    // g.solve();
    //
    // InexactRPCASolver b(data.blue, 0.05);
    // b.solve();
    //
    // TCVectorizedVidMat lowrank, sparse;
    // lowrank.rows = data.rows;
    // lowrank.cols = data.cols;
    // lowrank.nMat = data.nMat;
    // lowrank.size = data.size;
    // lowrank.fps = data.fps;
    // lowrank.ex = data.ex;
    // lowrank.red = r.getLowRank();
    // lowrank.green = g.getLowRank();
    // lowrank.blue = b.getLowRank();
    //
    // sparse.rows = data.rows;
    // sparse.cols = data.cols;
    // sparse.nMat = data.nMat;
    // sparse.size = data.size;
    // sparse.fps = data.fps;
    // sparse.ex = data.ex;
    // sparse.red = r.getSparse();
    // sparse.green = g.getSparse();
    // sparse.blue = b.getSparse();
    //
    // Converter::saveTCMat2MultiRGB("/Users/Marvin/Documents/mycode/cpp/rpca/video/lowrank/", lowrank);
    // Converter::saveTCMat2MultiRGB("/Users/Marvin/Documents/mycode/cpp/rpca/video/sparse/", sparse);
    Converter::mergeImgToVid("/Users/Marvin/Documents/mycode/cpp/rpca/imgs", "/Users/Marvin/Documents/mycode/cpp/rpca/result.avi", 15);
    Converter::mergeImgToVid("/Users/Marvin/Documents/mycode/cpp/rpca/result/lowrank", "/Users/Marvin/Documents/mycode/cpp/rpca/lowrank.avi", 15);
    Converter::mergeImgToVid("/Users/Marvin/Documents/mycode/cpp/rpca/result/sparse", "/Users/Marvin/Documents/mycode/cpp/rpca/sparse.avi", 15);


}
