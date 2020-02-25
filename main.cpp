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
void RGBFilter(Mat inputImage, Mat &outputImage);
void HSVFilter(Mat inputImage, Mat &outputImage);
//draw
void draw(Mat &image, vector<Point2f> points);
void getCrossPointAndIncrement(Vec4f LineA, Vec4f LineB, vector<Vertex> &vertexSet, int imgW, int imgH);
Point2f getCrossPoint(Point2f pa, Point2f pb);
Point2f getLinePoints(Vec4f line, vector<Point2f> &LinePoints);
//border detection
void borderHough(Mat inputImage, Mat &outputImage);
void borderContour(Mat inputImage, Mat &outputImage);
void lineCluster(vector<Point2f> kbPoints, vector<Point2f> &result);
void lineClusterVal(vector<Point2f> kbPoints, vector<Point2f> &result);
void pointCluster(vector<Point2f> intersections, vector<Point2f> &result);


int main() {

    Mat srcImage = imread("../images/yellowBorder13.jpg");
    namedWindow("原始图像", WINDOW_NORMAL);
    resizeWindow("原始图像", 1000, 1000);
    imshow("原始图像", srcImage);


    Mat dstImage = srcImage.clone();

    Mat mask = srcImage.clone();
    HSVFilter(srcImage, mask);
//    RGBFilter(srcImage, dstImage);

    borderHough(mask, dstImage);
//    borderContour(mask, dstImage);


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

    namedWindow("mask", WINDOW_NORMAL);
    resizeWindow("mask", 1000, 1000);
    imshow("mask", mask);

/*    //hough变换
    vector<Vec4f> lines;
    HoughLinesP(mask, lines, 1, CV_PI/180, 900, 500, 10);
    //延长直线
    for (unsigned int i = 0; i<lines.size(); i++)
    {
        Vec4f v = lines[i];
        lines[i][0] = 0;
        lines[i][1] = ((float)v[1] - v[3]) / (v[0] - v[2])* -v[0] + v[1]; //-kx0+y0=b
        lines[i][2] = inputImage.cols;
        lines[i][3] = ((float)v[1] - v[3]) / (v[0] - v[2])*(inputImage.cols - v[2]) + v[3];
    }

    Mat tempImage(mask.rows, mask.cols, CV_8U);
    //绘制直线
    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4f l = lines[i];
        line(tempImage, Point2f(l[0], l[1]), Point2f(l[2], l[3]), Scalar(255,255,255), 3, LINE_AA);
    }

    //形态学运算
    dilate(tempImage, tempImage, element); //膨胀
    //角点检测
    namedWindow("tempImage", WINDOW_NORMAL);
    resizeWindow("tempImage", 1000, 1000);
    imshow("tempImage", tempImage);*/
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
    //获取直线上的所有点
    vector<Point2f> LinePoints;
    vector<Point2f> kbPoints;
    vector<Point2f> kbPointsResult;
    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4f l = lines[i];
        line(tempImage, Point2f(l[0], l[1]), Point2f(l[2], l[3]), Scalar(0,255,0), 3, LINE_AA);

        float k = (l[3] - l[1]) / (l[2] - l[0]); //求出Line斜率
        float b = l[1] - k*l[0];
        kbPoints.push_back(Point2f(k, b));
//        getLinePoints(l, LinePoints);
    }
/*    cout << "**********input***********" << endl;
    for (int j = 0; j < kbPoints.size(); ++j) {
        cout << kbPoints[j].x << "," << kbPoints[j].y << endl;
    }

    lineCluster(kbPoints, kbPointsResult);
    cout << "**********output***********" << endl;
    for (int j = 0; j < kbPointsResult.size(); ++j) {
        cout << kbPointsResult[j].x << "," << kbPointsResult[j].y << endl;
    }*/



/*    //画出拟合后的线
    vector<Vec4f> linesCluster;
    Vec4f line_c;
    for (int i = 0; i < kbPointsResult.size(); ++i) {

        float k = kbPointsResult[i].x;
        float b = kbPointsResult[i].y;

        line_c[0] = 0;
        line_c[1] = b;
        line_c[2] = inputImage.cols;
        line_c[3] = k * inputImage.cols + b;

        linesCluster.push_back(line_c);

    }
    for (int i = 0; i < linesCluster.size(); ++i) {
        Vec4f l = linesCluster[i];
        line(tempImage, Point2f(l[0], l[1]), Point2f(l[2], l[3]), Scalar(0,255,0), 30, LINE_AA);

    }

    //存储拟合后线的交点
    vector<Point2f> intersection;
    for (unsigned int i = 0; i<kbPointsResult.size(); i++)
    {
        for (unsigned int j = i + 1; j<kbPointsResult.size(); j++)
        {
            Point2f pt = getCrossPoint(kbPointsResult[i], kbPointsResult[j]);
            if (pt.x >= 0 && pt.x < 5000 && pt.y >= 0)
            {
                intersection.push_back(pt);
            }
        }
    }

    //绘制交点
    cout << "************intersection**************" << endl;
    for (size_t i = 0; i < intersection.size(); i++)
    {
        cout << intersection[i].x << ", " << intersection[i].y << endl;
        circle(tempImage, intersection[i], 30, Scalar(0, 0, 255), -1);
    }*/


/*    //绘制hough直线上的点
    for (size_t i = 0; i < LinePoints.size(); i++)
    {
        cout << LinePoints[i].x << ", " << LinePoints[i].y << endl;
        circle(tempImage, LinePoints[i], 10, Scalar(0, 0, 255), -1);
    }*/

    int imgW = inputImage.cols;
    int imgH = inputImage.rows;

    cout << imgW << "," << imgH << endl;


    //存储hough直线的交点，并判断交点次数
    vector<Vertex> vertexSet;
    cout << "*********input**************" << endl;
    for (unsigned int i = 0; i<lines.size(); i++)
    {
        for (unsigned int j = i + 1; j<lines.size(); j++)
        {
            getCrossPointAndIncrement(lines[i], lines[j], vertexSet, imgW, imgH);

        }
    }
    cout << vertexSet.size() << endl;
    for (int i = 0; i < vertexSet.size(); ++i) {
        cout << "(" << vertexSet[i].x << ", " << vertexSet[i].y << ")" << endl;
//        circle(tempImage, Point(vertexSet[i].x, vertexSet[i].y), 30, Scalar(0, 0, 255), -1);
    }

    //找相交次数最多的4个点
    int max = 0;
    int maxUnder = -1;
    vector<Vertex> top4vertexSet;
    while (top4vertexSet.size() < 4) {
        max = 0;
        for (int i = 0; i < vertexSet.size(); i++) {
            if (vertexSet[i].crossTimes > max) {
                max = vertexSet[i].crossTimes;
                maxUnder = i;
            }
        }
        top4vertexSet.push_back(vertexSet[maxUnder]);
        vertexSet[maxUnder].crossTimes = -1;
    }

    cout << "11111111111111111" <<endl;
    for (int i = 0; i < top4vertexSet.size(); i++) {
        cout << "(" << top4vertexSet[i].x << "," << top4vertexSet[i].y <<")" << endl;
        circle(tempImage, Point(top4vertexSet[i].x, top4vertexSet[i].y), 30, Scalar(0, 0, 255), -1);
    }


/*    //点的聚类
    vector<Point2f> intersectionResult;
    pointCluster(intersection, intersectionResult);

    //绘制聚类后的交点
    for (size_t i = 0; i < intersectionResult.size(); i++)
    {
        circle(tempImage, intersectionResult[i], 10, Scalar(0, 0, 255), -1);
    }*/

    namedWindow("tempImage", WINDOW_NORMAL);
    resizeWindow("tempImage", 1000, 1000);
    imshow("tempImage", tempImage);





/*    vector<Point2f> points;
    for (int i = 0; i < inputImage.rows; ++i) {
        for (int j = 0; j < inputImage.cols; ++j) {
            if(inputImage.at<uchar>(i, j) == 255)
                points.push_back(Point2f(j, i));
        }
    }

    draw(outputImage, points);*/
}

void pointCluster(vector<Point2f> intersections, vector<Point2f> &result){
    int sampleCount = intersections.size();
    int clusterCount = 4;
    Mat points(sampleCount, 1, CV_32FC2);
    Mat labels;
    Mat centers(clusterCount, 1, points.type());

    for (int row = 0; row < intersections.size(); row++)
    {
        points.at<Point2f>(row) = intersections[row];
    }

    TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
    kmeans(points, clusterCount, labels, criteria, 1000, KMEANS_PP_CENTERS, centers);
    cout << "*********output**************" << endl;
    for (int i = 0; i < centers.rows; i++)
    {
        Point2f pt = centers.at<Point2f>(i);
        result.push_back(pt);
        cout << "(" << pt.x << "," << pt.y << ")" << endl;
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

Point2f getCrossPoint(Point2f pa, Point2f pb){
    Point2f crossPoint;
    crossPoint.x = (pb.y - pa.y) / (pa.x - pb.x);
    crossPoint.y = (pa.x * pb.y - pb.x * pa.y) / (pa.x - pb.x);
    return crossPoint;
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

void RGBFilter(Mat inputImage, Mat &outputImage){
    int rowNumber = inputImage.rows;
    int colNumber = inputImage.cols;

    vector<Point2f> points;

    for (int i = 0; i < rowNumber; ++i) {
        for (int j = 0; j < colNumber; ++j) {
            if ((inputImage.at<Vec3b>(i, j)[0] > 30 && inputImage.at<Vec3b>(i, j)[0] < 70)
                && (inputImage.at<Vec3b>(i, j)[1] > 90 && inputImage.at<Vec3b>(i, j)[1] < 260)
                && (inputImage.at<Vec3b>(i, j)[2] > 215 && inputImage.at<Vec3b>(i, j)[2] < 260)) {
                points.push_back(Point2f(j, i));
            }
        }
    }

    draw(outputImage, points);
}


void borderContour(Mat inputImage, Mat &outputImage){
    vector<vector<Point>> contours;
    findContours(inputImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    //冒泡排序，由小到大排序
    vector<Point> vptmp;
    for(int i=1;i<contours.size();i++){
        for(int j=contours.size()-1;j>=i;j--){
            if(contours[j].size()<contours[j-1].size()){
                vptmp = contours[j-1];
                contours[j-1] = contours[j];
                contours[j] = vptmp;
            }
        }
    }

    Mat contoursMat = Mat::zeros(inputImage.size(),CV_8UC3);
    drawContours(contoursMat, contours, contours.size()-1, Scalar(0, 0, 255), 2, 8);

    namedWindow("contours", WINDOW_NORMAL);
    resizeWindow("contours", 1000, 1000);
    imshow("contours", contoursMat);

    vector<vector<Point>> lines(contours.size());
    for (int i = 0; i < contours.size(); i++) {
        //拟合
        approxPolyDP(contours[i],lines[i],9,true);
    }
    drawContours(outputImage, lines, -1,Scalar(0, 255, 0),20,8 );
}

void draw(Mat &image, vector<Point2f> points){

    RotatedRect box = minAreaRect(points);
    Point2f vertex[4];
    box.points(vertex);

    for (int i = 0; i < 4; ++i) {
        circle(image, vertex[i], 50, Scalar(0, 0, 255), -1);
        line(image,vertex[i], vertex[(i+1)%4], Scalar(0, 255, 0), 30);
    }
}

void lineClusterVal(vector<Point2f> kbPoints, vector<Point2f> &result){

    vector<Point2f> p1, p2, p3, p4;
    for (int i = 0; i < kbPoints.size(); ++i) {
        if(kbPoints[i].x < 0)
            p1.push_back(kbPoints[i]);
        else if(kbPoints[i].x < 1)
            p2.push_back(kbPoints[i]);
        else if(kbPoints[i].x < 2)
            p3.push_back(kbPoints[i]);
        else
            p4.push_back(kbPoints[i]);
    }
    vector<vector<Point2f>> pt;
    pt.push_back(p1);
    pt.push_back(p2);
    pt.push_back(p3);
    pt.push_back(p4);

    for (int i = 0; i < pt.size(); ++i) {
        float k_sum = 0, b_sum = 0, k_mean = 0, b_mean = 0;
        for (int j = 0; j < pt[i].size(); ++j) {
            k_sum += pt[i][j].x;
            b_sum += pt[i][j].y;
        }
        k_mean = k_sum / pt[i].size();
        b_mean = b_sum / pt[i].size();
        result.push_back(Point2f(k_mean, b_mean));
    }

    cout << result.size() << endl;
}
