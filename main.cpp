#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using namespace std;
using namespace cv;

struct Vertex {
    int x;
    int y;
    int crossTimes = 0;
    Vertex(int posX, int posY): x(posX), y(posY) {}
    void setXY(int _x, int _y) {
        x = _x;
        y = _y;
    }
    void addCrossTimes() {
        crossTimes++;
    }
};

//色相（黄色）
int hmin = 26;
int hmax = 34;

//饱和度
int smin = 43;
int smax = 255;

//亮度
int vmin = 46;
int vmax = 255;

//image process
void HSVFilter(Mat inputImage, Mat &outputImage);
//border detection
void borderHough(Mat inputImage, Mat &outputImage);
void getCrossPointAndIncrement(Vec4f LineA, Vec4f LineB, vector<Vertex> &vertexSet, int imgW, int imgH);
void drawLines(vector<Point> top4vertexSet, Mat &outputImage);
//others
void lineCluster(vector<Point2f> kbPoints, vector<Point2f> &result);
Point2f getLinePoints(Vec4f line, vector<Point2f> &LinePoints);



int main() {

    Mat srcImage = imread("../images/yellowBorder17.jpg");
    namedWindow("原始图像", WINDOW_NORMAL);
    resizeWindow("原始图像", 1000, 1000);
    imshow("原始图像", srcImage);


    Mat dstImage = srcImage.clone();

    Mat mask = srcImage.clone();
    HSVFilter(srcImage, mask);

    borderHough(mask, dstImage);


    namedWindow("目标图像", WINDOW_NORMAL);
    resizeWindow("目标图像", 1000, 1000);
    imshow("目标图像", dstImage);
    waitKey(0);
    return 0;
}

void HSVFilter(Mat inputImage, Mat &outputImage){
    Mat hsvImage;
    //bgr转hsv
    cvtColor(inputImage, hsvImage, CV_BGR2HSV);
    Mat mask;

    //二值化
    inRange(hsvImage, Scalar(hmin, smin, vmin), Scalar(hmax, smax, vmax), mask); //scalar不是bgr吗，为什么可以限定上下限

    //形态学运算
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    erode(mask, mask, element); //腐蚀
    dilate(mask, mask, element); //膨胀

    outputImage = mask.clone();

/*    namedWindow("mask", WINDOW_NORMAL);
    resizeWindow("mask", 1000, 1000);
    imshow("mask", mask);*/
}

void borderHough(Mat inputImage, Mat &outputImage){
    vector<Vec4f> lines;
    HoughLinesP(inputImage, lines, 1, CV_PI/180, 900, 500, 10);  //LSD算法kk

    cout << "共检测到原始直线" << lines.size() << "条" << endl;

/*    // 延长直线
    for (unsigned int i = 0; i<lines.size(); i++)
    {
        Vec4f v = lines[i];
        lines[i][0] = 0;
        lines[i][1] = ((float)v[1] - v[3]) / (v[0] - v[2])* -v[0] + v[1]; //-kx0+y0=b
        lines[i][2] = inputImage.cols;
        lines[i][3] = ((float)v[1] - v[3]) / (v[0] - v[2])*(inputImage.cols - v[2]) + v[3];
    }*/

    Mat tempImage = outputImage.clone();

    //绘制直线
    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4f l = lines[i];
        line(tempImage, Point2f(l[0], l[1]), Point2f(l[2], l[3]), Scalar(0,255,0), 3, LINE_AA);
    }

/*    namedWindow("tempImage", WINDOW_NORMAL);
    resizeWindow("tempImage", 1000, 1000);
    imshow("tempImage", tempImage);*/

    int imgW = inputImage.cols;
    int imgH = inputImage.rows;

    //存储hough直线的交点，并判断交点相交次数
    vector<Vertex> vertexSet;
    for (unsigned int i = 0; i<lines.size(); i++)
    {
        for (unsigned int j = i + 1; j<lines.size(); j++)
        {
            getCrossPointAndIncrement(lines[i], lines[j], vertexSet, imgW, imgH);

        }
    }

/*    //绘制所有交点
    for (int i = 0; i < vertexSet.size(); ++i) {
        cout << "(" << vertexSet[i].x << ", " << vertexSet[i].y << ")" << endl;
        circle(tempImage, Point(vertexSet[i].x, vertexSet[i].y), 30, Scalar(0, 0, 255), -1);
    }*/

    //找相交次数最多的4个点
    int max = 0;
    int maxUnder = -1;
    vector<Point> top4vertexSet;
    while (top4vertexSet.size() < 4) {
        max = 0;
        for (int i = 0; i < vertexSet.size(); i++) {
            if (vertexSet[i].crossTimes > max) {
                max = vertexSet[i].crossTimes;
                maxUnder = i;
            }
        }
        top4vertexSet.push_back(Point(vertexSet[maxUnder].x, vertexSet[maxUnder].y));
        vertexSet[maxUnder].crossTimes = -1;
    }
    //绘制直线
    drawLines(top4vertexSet, outputImage);
    //绘制最终的四个交点
    for (int i = 0; i < top4vertexSet.size(); i++) {
        cout << "(" << top4vertexSet[i].x << "," << top4vertexSet[i].y <<")" << endl;
        circle(outputImage, top4vertexSet[i], 40, Scalar(0, 0, 255), -1);
    }

}

//求交点函数，并对交点做交叉次数做累加
void getCrossPointAndIncrement(Vec4f LineA, Vec4f LineB, vector<Vertex> &vertexSet, int imgW, int imgH)
{
    float ka, kb;
    ka = (LineA[3] - LineA[1]) / (LineA[2] - LineA[0]); //求出LineA斜率
    kb = (LineB[3] - LineB[1]) / (LineB[2] - LineB[0]); //求出LineB斜率

    Point2f crossPoint;
    crossPoint.x = (ka*LineA[0] - LineA[1] - kb*LineB[0] + LineB[1]) / (ka - kb);
    crossPoint.y = (ka*kb*(LineA[0] - LineB[0]) + ka*LineB[1] - kb*LineA[1]) / (ka - kb);

    int x = (int)(round)(crossPoint.x);
    int y = (int)(round)(crossPoint.y);

    int VertexGap = 10000;


    if (x >= 0 && x <= imgW && y >= 0 && y <= imgH) {  //在图像区域内
        int i = 0;
        for (i = 0; i < vertexSet.size(); i++) {
            int oldX = vertexSet[i].x;
            int oldY = vertexSet[i].y;

            //附近有特别靠近的点，可以合并
            if ((oldX - x) * (oldX - x) + (oldY - y) * (oldY - y) <= VertexGap) {
                vertexSet[i].addCrossTimes();
                break;
            }
        }

        if (i == vertexSet.size()) {  //如果该点附近没有距离特别近的点，自身作为一个新点
            Vertex newVertex(x, y);
            vertexSet.push_back(newVertex);
        }
    }


}

//对角线以外的两个点，一个在对角线上，一个在对角线下
void drawLines(vector<Point> top4vertexSet, Mat &outputImage){
    int crossPoint = 0;
    for (int i = 1; i < 4; i++) {   //第0个点与第i个点连线
        double temp_k = (double)(top4vertexSet[i].y - top4vertexSet[0].y) / (double)(top4vertexSet[i].x - top4vertexSet[0].x);
        double temp_b = (double)top4vertexSet[0].y - temp_k * (double)top4vertexSet[0].x;

        int flag = 0;  //标志为正还是为负
        for (int j = 1; j < 4; j++) {
            if (j != i) {
                //第j个点的y坐标减线上坐标
                double diff = (double)top4vertexSet[j].y - (temp_k * (double)top4vertexSet[j].x + temp_b);
                if (flag == 0) {
                    flag = diff > 0 ? 1 : -1;
                }
                else {
                    if (flag == 1 && diff <= 0 || flag == -1 && diff > 0) {
                        crossPoint = i;
                        break;
                    }
                }
            }
        }
        if (crossPoint != 0)
            break;
    }

    for (int i = 1; i < 4; i++) {
        if (i != crossPoint) {
            line(outputImage, top4vertexSet[i], top4vertexSet[0], Scalar(0,255,0), 30, LINE_AA);
            line(outputImage, top4vertexSet[i], top4vertexSet[crossPoint], Scalar(0,255,0), 30, LINE_AA);
        }
    }
}






//获取直线上的点的函数
/*函数功能：获取霍夫变换得到的直线上的点*/
/*输入：Vec4f类型直线*/
/*返回：Point2f类型的点集*/
Point2f getLinePoints(Vec4f line, vector<Point2f> &LinePoints){
    float x1 = line[0];
    float y1 = line[1];
    float x2 = line[2];
    float y2 = line[3];

    float delta_x = x1 - x2;
    float delta_y = y1 - y2;

/*    int maxstep;
    maxstep = abs(delta_x)>abs(delta_y) ? abs(delta_x):abs(delta_y);*/

    int minstep;
    minstep = abs(delta_x)>abs(delta_y) ? abs(delta_y):abs(delta_x);

/*    float xUnitstep = abs(delta_x) / maxstep;
    float yUnitstep = abs(delta_y) / maxstep;*/

    float xUnitstep = -delta_x / minstep;
    float yUnitstep = -delta_y / minstep;


    float x = x1, y = y1;
    LinePoints.push_back(Point2f(x, y));

    for(int j=0; j<minstep; ++j){
        x = x + xUnitstep;
        y = y + yUnitstep;

        LinePoints.push_back(Point2f(x, y));
    }


}

void lineCluster(vector<Point2f> kbPoints, vector<Point2f> &result){
    //初始化
/*    int width = 2;
    int height = kbPoints.size();*/
    int sampleCount = kbPoints.size();
    int clusterCount = 4;
    Mat points(sampleCount, 1, CV_32FC2);
    Mat labels;
    Mat centers(clusterCount, 1, points.type());

    // 将数据类型转化到样本数据
/*    int index = 0;
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            // 多维转一维
            index = row * width + col;
            points.at<float>(index, 0) = kbPoints[index].x;
            points.at<float>(index, 1) = kbPoints[index].y;
        }
    }*/


    for (int row = 0; row < kbPoints.size(); row++)
    {
        points.at<Point2f>(row) = kbPoints[row];
    }

    TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 1000, 0.01);
    kmeans(points, clusterCount, labels, criteria, 1000, KMEANS_PP_CENTERS, centers);


/*    for (int i = 0; i < centers.rows; i++)
    {
        int x = centers.at<float>(i, 0);
        int y = centers.at<float>(i, 1);
        result.push_back(Point2f(x, y));
    }*/

    for (int i = 0; i < centers.rows; i++)
    {
        Point2f pt = centers.at<Point2f>(i);
        result.push_back(pt);
    }


}


