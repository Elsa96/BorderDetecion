#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using namespace std;
using namespace cv;

//几个定值参数要修改：1. 两点间的距离小于多少合并为一个点  2. 相交次数大于几倍为三条直线  3. 画面边界往里收缩多少

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
void drawLines(vector<Vertex> top4vertexSet, Mat &outputImage);
void drawBox(vector<Vertex> vertexSet, Mat &outputImage);
void drawPoints(vector<Vertex> vertexSet, Mat &outputImage);
//special case
void mostIntersections(vector<Vec4f> lines, vector<Vertex> &topVertexSet, int topVertexNum, int imgW, int imgH);\
void pointColor(Mat image, vector<Vertex> inputVertexSet, vector<Vertex> &outputVertexSet);
//others
void lineCluster(vector<Point2f> kbPoints, vector<Point2f> &result);
Point2f getLinePoints(Vec4f line, vector<Point2f> &LinePoints);


int main() {
    Mat srcImage = imread("../images/yellowBorder21.jpg");
/*    namedWindow("原始图像", WINDOW_NORMAL);
    resizeWindow("原始图像", 1000, 1000);
    imshow("原始图像", srcImage);*/


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
    HoughLinesP(inputImage, lines, 1, CV_PI/180, 900, 500, 10);  //第五个参数：超过这个值才被检测出直线

    cout << "共检测到原始直线" << lines.size() << "条" << endl;

    int imgW = inputImage.cols;
    int imgH = inputImage.rows;

    cout << "图片的宽为" << imgW << endl;
    cout << "图片的高为" << imgH << endl;
    cout << "************************************" << endl;

    //TODO 如何精简
    int gap = 50;
    //上下左右四条边
    Vec4f lineUp(gap,gap,imgW-gap,gap);//up
    Vec4f lineDown(gap,imgH-gap,imgW-gap,imgH-gap);//down
    Vec4f lineLeft(gap+10,gap,gap,imgH-gap);//left 注意不要完全垂直
    Vec4f lineRight(imgW-gap-10,gap,imgW-gap,imgH-gap);//right

    //将画面上下两条边加进去
    int linesNum = 20;  //TODO 参数linesNum
    vector<Vec4f> lines3SidesUD(lines);
    lines3SidesUD.insert(lines3SidesUD.end(), linesNum, lineUp);
    lines3SidesUD.insert(lines3SidesUD.end(), linesNum, lineDown);
    //将画面左右两条边加进去
    vector<Vec4f> lines3SidesLR(lines);
    lines3SidesLR.insert(lines3SidesLR.end(), linesNum, lineLeft);
    lines3SidesLR.insert(lines3SidesLR.end(), linesNum, lineRight);
    //将画面上下左右都加进去
    vector<Vec4f> lines2Sides(lines3SidesUD);
    lines2Sides.insert(lines2Sides.end(), linesNum, lineLeft);
    lines2Sides.insert(lines2Sides.end(), linesNum, lineRight);
    cout << "直线:" << lines2Sides.size() << "条" << endl;

    Mat tempImage = outputImage.clone();
    //    Mat tempImage(outputImage.rows, outputImage.cols, CV_8UC1);


/*    // 延长直线
    for (unsigned int i = 0; i<lines.size(); i++)
    {
        Vec4f v = lines[i];
        lines[i][0] = 0;
        lines[i][1] = ((float)v[1] - v[3]) / (v[0] - v[2])* -v[0] + v[1]; //-kx0+y0=b
        lines[i][2] = inputImage.cols;
        lines[i][3] = ((float)v[1] - v[3]) / (v[0] - v[2])*(inputImage.cols - v[2]) + v[3];
    }*/

/*    //绘制直线
    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4f l = lines[i];
        line(outputImage, Point2f(l[0], l[1]), Point2f(l[2], l[3]), Scalar(0,255,0), 3, LINE_AA);
    }*/

    vector<Vertex> top4vertexSet;
    mostIntersections(lines, top4vertexSet, 4, imgW, imgH);

    //绘制相交次数最多的六个交点
    cout << "交点坐标和相交次数：" << endl;
    for (int i = 0; i < top4vertexSet.size(); i++) {
        cout << "(" << top4vertexSet[i].x << "," << top4vertexSet[i].y <<")" << top4vertexSet[i].crossTimes << endl;
    }

    //判断有几条直线
    if(top4vertexSet[0].crossTimes > 4 * top4vertexSet[1].crossTimes){  //TODO 参数4
        vector<Vertex> top9vertexSet;
        cout << "只有两条直线，并相交" << endl;
        mostIntersections(lines2Sides, top9vertexSet, 9, imgW, imgH);
//        drawPoints(top9vertexSet, outputImage);
        vector<Vertex> vertexResult;
        pointColor(outputImage, top9vertexSet, vertexResult);
        cout << "交点个数：" << vertexResult.size() << endl;
        //TODO 还需考虑与边界相交的两个点在同一条边界的情况
        int findPoint4 = 0;
        int draw = 0;
        for (int i = 0; i < vertexResult.size(); ++i) {
            if(vertexResult[i].x < 2 *gap){
                for (int j = 0; j < vertexResult.size(); ++j) {
                    if(j != i){
                        if(vertexResult[j].x < 2 * gap){
                            drawBox(vertexResult, outputImage);
                            findPoint4 = 1;
                            draw = 1;
                            break;
                        }
                        if(vertexResult[j].y < 2 * gap){
                            Vertex newVertex(gap, gap);
                            vertexResult.push_back(newVertex);
                            findPoint4 = 1;
                            break;
                        }
                        if(vertexResult[j].y > imgH - 2 * gap) {
                            Vertex newVertex(gap, imgH - gap);
                            vertexResult.push_back(newVertex);
                            findPoint4 = 1;
                            break;
                        }
                    }
                }
            }
            if(vertexResult[i].x > imgW - 2 * gap){
                for (int j = 0; j < vertexResult.size(); ++j) {
                    if(j != i){
                        if(vertexResult[j].x > imgW - 2 * gap){
                            drawBox(vertexResult, outputImage);
                            findPoint4 = 1;
                            draw = 1;
                            break;
                        }
                        if(vertexResult[j].y < 2 * gap){
                            Vertex newVertex(imgW - gap, gap);
                            vertexResult.push_back(newVertex);
                            findPoint4 = 1;
                            break;
                        }
                        if(vertexResult[j].y > imgH - 2 * gap) {
                            Vertex newVertex(imgW - gap, imgH - gap);
                            vertexResult.push_back(newVertex);
                            findPoint4 = 1;
                            break;
                        }
                    }
                }
            }
            if((vertexResult[i].y > imgH - 2 * gap) &&
            (vertexResult[(i+1)%vertexResult.size()].y > imgH - 2 * gap)){
                drawBox(vertexResult, outputImage);
                draw = 1;
                break;
            }

            if((vertexResult[i].y < 2 *gap) &&
            (vertexResult[(i+1)%vertexResult.size()].y < 2 *gap)){
                drawBox(vertexResult, outputImage);
                draw = 1;
                break;
            }
            if(findPoint4 == 1)
                break;
        }
        if(draw == 0){
            drawLines(vertexResult, outputImage);
            drawPoints(vertexResult, outputImage);
        }
    }
    else if(top4vertexSet[0].crossTimes < 1000){  //TODO 参数1000
        cout << "只有两条直线，平行" << endl;
        vector<Vertex> top4vertexSet_2;
        mostIntersections(lines2Sides, top4vertexSet_2, 4, imgW, imgH);
        if(top4vertexSet_2[0].crossTimes <= 4 * top4vertexSet_2[2].crossTimes){  //排除只有一条边的情况
            drawLines(top4vertexSet_2, outputImage);
            drawPoints(top4vertexSet_2, outputImage);
        }

    }
    else if(top4vertexSet[0].crossTimes > 4 * top4vertexSet[2].crossTimes){  //TODO 参数4
        cout << "只有三条直线" << endl;
        vector<Vertex> top6vertexSet;
        int yGap = 500;  //TODO 参数yGap
        if(abs(top4vertexSet[0].y - top4vertexSet[1].y)< yGap) //若两交点y很接近，则取上下边界
            mostIntersections(lines3SidesUD, top6vertexSet, 6, imgW, imgH);
        else
            mostIntersections(lines3SidesLR, top6vertexSet, 6, imgW, imgH);

        vector<Vertex> vertexResult;
        pointColor(outputImage, top6vertexSet, vertexResult);

        cout << "交点个数为：" << vertexResult.size() << endl;
        //绘制直线
        drawLines(vertexResult, outputImage);
        //绘制交点
        drawPoints(vertexResult, outputImage);
    }
    else{
        cout << "四条直线" << endl;
        //绘制直线
        drawLines(top4vertexSet, outputImage);
        //绘制最终的四个交点
        drawPoints(top4vertexSet, outputImage);
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

    int VertexGap = 40000; //TODO VerTexGap

    if (x >= -imgW/2 && x <= imgW * 1.5 && y >= -imgH/2 && y <= imgH * 1.5) {  //在图像区域内
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
void drawLines(vector<Vertex> top4vertexSet, Mat &outputImage){
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
            line(outputImage, Point(top4vertexSet[i].x, top4vertexSet[i].y), Point(top4vertexSet[0].x, top4vertexSet[0].y), Scalar(0,255,0), 30, LINE_AA);
            line(outputImage, Point(top4vertexSet[i].x, top4vertexSet[i].y), Point(top4vertexSet[crossPoint].x, top4vertexSet[crossPoint].y), Scalar(0,255,0), 30, LINE_AA);
        }
    }
}

void drawPoints(vector<Vertex> vertexSet, Mat &outputImage){
    for (int i = 0; i < vertexSet.size(); i++) {
        cout << "(" << vertexSet[i].x << "," << vertexSet[i].y <<")" << vertexSet[i].crossTimes << endl;
        circle(outputImage, Point(vertexSet[i].x, vertexSet[i].y), 30, Scalar(0, 0, 255), -1);
    }
}

void drawBox(vector<Vertex> vertexSet, Mat &outputImage){

    for (int i = 0; i < vertexSet.size(); i++) {
        Point pt = Point(vertexSet[i].x, vertexSet[i].y);
        Point pt1 = Point(vertexSet[(i + 1) % 3].x, vertexSet[(i + 1) % 3].y);
        line(outputImage, pt, pt1, Scalar(0, 255, 0), 30);
    }
    for (int i = 0; i < vertexSet.size(); ++i) {
        Point pt = Point(vertexSet[i].x, vertexSet[i].y);
        circle(outputImage, pt, 30, Scalar(0, 0, 255), -1);
    }
}

void mostIntersections(vector<Vec4f> lines, vector<Vertex> &topVertexSet, int topVertexNum, int imgW, int imgH){
    //获取所有直线的交点和相交次数
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
        cout << "(" << vertexSet[i].x << ", " << vertexSet[i].y << ")" << vertexSet[i].crossTimes << endl;
        circle(outputImage, Point(vertexSet[i].x, vertexSet[i].y), 10, Scalar(0, 0, 255), -1);
    }*/

    //找相交次数最多的topVertexNum个点
    int max = 0;
    int maxUnder = -1;
    while (topVertexSet.size() < topVertexNum) {
        max = 0;
        for (int i = 0; i < vertexSet.size(); i++) {
            if (vertexSet[i].crossTimes > max) {
                max = vertexSet[i].crossTimes;
                maxUnder = i;
            }
        }
        topVertexSet.push_back(vertexSet[maxUnder]);
        vertexSet[maxUnder].crossTimes = -1;
    }
}

void pointColor(Mat image, vector<Vertex> inputVertexSet, vector<Vertex> &outputVertexSet){
    int imgW = image.cols;
    int imgH = image.rows;

    Mat hsvImage;
    cvtColor(image, hsvImage, CV_BGR2HSV); //bgr转hsv

    for (int i = 0; i < inputVertexSet.size(); ++i) {
        int x = inputVertexSet[i].x;
        int y = inputVertexSet[i].y;
        //TODO 如何精简
        if(x < 0)
            inputVertexSet[i].x = 0;
        if(x > imgW)
            inputVertexSet[i].x = imgW;
        if(y < 0)
            inputVertexSet[i].y = 0;
        if(y > imgH)
            inputVertexSet[i].y = imgH;
        int range = 200; //TODO 参数range
        int flag = 0;
        //注意不要超出画面边界
        int xMin = x-range;
        int xMax = x+range;
        int yMin = y-range;
        int yMax = y+range;
        if(xMin < 0)
            xMin = 0;
        if(xMin > imgW)
            xMin = imgW-10;
        if(xMax < 0)
            xMax = 10;
        if(xMax > imgW)
            xMax = imgW;
        if(yMin < 0)
            yMin = 0;
        if(yMin > imgH)
            yMin = imgH-10;
        if(yMax < 0)
            yMax = 10;
        if(yMax > imgH)
            yMax = imgH;
        //看交点是否为黄色
        for (int j = xMin; j < xMax ; j = j+5) {
            for (int k = yMin; k < yMax; k = k+5) {
                int h = hsvImage.at<Vec3b>(k, j)[0];
                int s = hsvImage.at<Vec3b>(k, j)[1];
                int v = hsvImage.at<Vec3b>(k, j)[2];
                if((h > hmin && h < hmax) &&
                   (s > smin && s < smax) &&
                   (v > vmin && v < vmax)){
                    outputVertexSet.push_back(inputVertexSet[i]);
                    flag = 1;
                    break;
                }
            }
            if(flag == 1)
                break;
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
