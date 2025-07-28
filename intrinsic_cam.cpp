#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace std;

int main() {
    Size boardSize(4, 4); // number of inner corners
    float squareSize = 0.110; // in meters

    vector<vector<Point3f>> objpoints;
    vector<vector<Point2f>> imgpoints;

    vector<Point3f> objp;
    for (int i = 0; i < boardSize.height; ++i)
        for (int j = 0; j < boardSize.width; ++j)
            objp.emplace_back(j * squareSize, i * squareSize, 0);

    string path = "data/july8/cam1_cal_data/*.png";
    vector<String> imagePaths;
    glob(path, imagePaths);

    for (size_t i = 0; i < imagePaths.size(); i++) {
        Mat img = imread(imagePaths[i]);
        if (img.empty()) {
            cerr << "Failed to load " << imagePaths[i] << endl;
            continue;
        }

        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);

        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, boardSize, corners);

        if (found) {
            cornerSubPix(gray, corners, Size(15,15), Size(-1,-1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001));
            drawChessboardCorners(img, boardSize, corners, found);

            imgpoints.push_back(corners);
            objpoints.push_back(objp);

            imshow("Chessboard", img);
            waitKey(100);
        }
    }

    destroyAllWindows();

    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;

    calibrateCamera(objpoints, imgpoints, Size(640,480), cameraMatrix, distCoeffs, rvecs, tvecs);

    cout << "Camera Matrix:\n" << cameraMatrix << endl;
    cout << "Distortion Coefficients:\n" << distCoeffs << endl;

    return 0;
}