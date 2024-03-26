#pragma once
#include <iostream>
#include <cmath>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

int Otsu(Mat matSrc)
{
	if (CV_8UC1 != matSrc.type())
		return -1;
	int nCols = matSrc.cols;
	int nRows = matSrc.rows;
	int nPixelNum = nCols * nRows;
	// 初始化
	int pixelNum[256];
	double probability[256];
	for (int i = 0; i < 256; i++)
	{
		pixelNum[i] = 0;
		probability[i] = 0.0;
	}
	// 统计像素数和频率
	for (int j = 0; j < nRows; j++)
	{
		for (int i = 0; i < nCols; i++)
		{
			pixelNum[matSrc.at<uchar>(j, i)]++;
		}
	}
	for (int i = 0; i < 256; i++)
	{
		probability[i] = (double)0.1 * pixelNum[i] / nPixelNum;
	}
	// 计算
	int nThreshold = 0;			 // 最佳阈值
	double dMaxDelta = 0.0;		 // 最大类间方差
	double dMean_0 = 0.0;		 // 左边平均值
	double dMean_1 = 0.0;		 // 右边平均值
	double dDelta = 0.0;		 // 类间方差
	double dMean_0_temp = 0.0;	 // 左边平均值中间值
	double dMean_1_temp = 0.0;	 // 右边平均值中间值
	double dProbability_0 = 0.0; // 左边频率值
	double dProbability_1 = 0.0; // 右边频率值
	for (int j = 0; j < 256; j++)
	{
		for (int i = 0; i < 256; i++)
		{
			if (i < j) // 前半部分
			{
				dProbability_0 += probability[i];
				dMean_0_temp += i * probability[i];
			}
			else // 后半部分
			{
				dProbability_1 += probability[i];
				dMean_1_temp += i * probability[i];
			}
		}
		// 计算平均值
		// fMean_0_teamp计算的是前半部分的灰度值的总和除以总像素数，
		// 所以要除以前半部分的频率才是前半部分的平均值，后半部分同样
		dMean_0 = dMean_0_temp / dProbability_0;
		dMean_1 = dMean_1_temp / dProbability_1;
		dDelta = (double)(dProbability_0 * dProbability_1 * pow((dMean_0 - dMean_1), 2));
		if (dDelta > dMaxDelta)
		{
			dMaxDelta = dDelta;
			nThreshold = j;
		}
		// 相关参数归零
		dProbability_0 = 0.0;
		dProbability_1 = 0.0;
		dMean_0_temp = 0.0;
		dMean_1_temp = 0.0;
		dMean_0 = 0.0;
		dMean_1 = 0.0;
		dDelta = 0.0;
	}
	return nThreshold;
}
// 获取水平投影的数组信息
int *getHorPrjArr(const Mat binaryImage)
{
	// Mat hpImage(binaryImage.rows, binaryImage.cols, CV_8UC1);
	// hpImage.setTo(255);
	int *h = new int[binaryImage.rows]();
	for (int y = 0; y < (int)binaryImage.rows; y++)
	{
		for (int x = 0; x < (int)binaryImage.cols; x++)
		{
			if ((int)binaryImage.at<uchar>(y, x) == 0)
				h[y]++;
		}
	}
	return h;
}

int *getVerPrjArr(const Mat binaryImage)
{
	// Mat hpImage(binaryImage.rows, binaryImage.cols, CV_8UC1);
	// hpImage.setTo(255);
	int *v = new int[binaryImage.cols]();
	for (int x = 0; x < (int)binaryImage.cols; x++)
	{
		for (int y = 0; y < (int)binaryImage.rows; y++)
		{
			if ((int)binaryImage.at<uchar>(y, x) == 0)
				v[x]++;
		}
	}
	return v;
}

Mat getHorPrjMat(const Mat binaryImage)
{
	Mat hpImage(binaryImage.rows, binaryImage.cols, CV_8UC1);
	hpImage.setTo(255);
	int *h = new int[binaryImage.rows]();
	for (int y = 0; y < (int)binaryImage.rows; y++)
	{
		for (int x = 0; x < (int)binaryImage.cols; x++)
		{
			if ((int)binaryImage.at<uchar>(y, x) == 0)
				h[y]++;
		}
	}
	for (int y = 0; y < (int)binaryImage.rows; y++)
	{
		for (int x = 0; x < h[y]; x++)
		{
			hpImage.col((int)binaryImage.cols - x - 1).row(y) = 0;
		}
	}
	delete[] h;
	return hpImage;
}

Mat getVerPrjMat(const Mat binaryImage)
{
	Mat hpImage(binaryImage.rows, binaryImage.cols, CV_8UC1);
	hpImage.setTo(255);
	int *v = new int[binaryImage.cols]();
	for (int x = 0; x < (int)binaryImage.cols; x++)
	{
		for (int y = 0; y < (int)binaryImage.rows; y++)
		{
			if ((int)binaryImage.at<uchar>(y, x) == 0)
				v[x]++;
		}
	}

	for (int x = 0; x < (int)binaryImage.cols; x++)
	{
		for (int y = 0; y < v[x]; y++)
		{
			hpImage.row((int)binaryImage.rows - y - 1).col(x) = 0;
		}
	}

	delete[] v;
	return hpImage;
}
// 根据水平投影数组信息来分割，最开头部分即为兴趣区域
vector<Mat> splitImageVer(Mat srcImage)
{
	vector<Mat> result;
	int *v = getVerPrjArr(srcImage);
	int leftLoc = 0, rightLoc = 0, blankFlag = 0, noiseLength = 7;
	for (int y = 0; y < (int)srcImage.cols; y++)
	{
		if (!blankFlag)
		{
			if (v[y])
			{
				leftLoc = y;
				blankFlag = 1;
			}
		}
		else
		{
			if (!v[y] || y == (int)srcImage.cols - 1)
			{
				rightLoc = y;
				blankFlag = 0;
				// if(rightLoc-leftLoc+1>noiseLength)
				result.push_back(Mat(srcImage, Rect(leftLoc, 0, rightLoc - leftLoc + 1, (int)srcImage.rows - 1)));
			}
		}
	}
	/*if((int)result.size()>4)
		result.erase(result.begin(), result.begin() + 4);*/
	/*for (int i = 0; i < (int)result.size(); i++)
	{
		imshow(to_string(i), result[i]);
		moveWindow(to_string(i), 60 * (i + 1), 100);
	}*/
	delete[] v;
	return result;
}

vector<Mat> splitImageHor(Mat srcImage)
{
	// 是否为白色或者黑色根据二值图像的处理得来
	Mat painty(srcImage.size(), CV_8UC1, Scalar(255)); // 初始化为全白

	// 水平投影
	int *pointcount = new int[srcImage.rows];	   // 在二值图片中记录行中特征点的个数
	memset(pointcount, 0, (int)srcImage.rows * 4); // 注意这里需要进行初始化

	for (int i = 0; i < srcImage.rows; i++)
		for (int j = 0; j < srcImage.cols; j++)
			if (srcImage.at<uchar>(i, j) == 0)
				pointcount[i]++; // 记录每行中黑色点的个数 //水平投影按行在y轴上的投影

	for (int i = 0; i < srcImage.rows; i++)
		for (int j = 0; j < pointcount[i]; j++) // 根据每行中黑色点的个数，进行循环
			painty.at<uchar>(i, j) = 0;

	// imshow("painty", painty);

	vector<Mat> result;
	int startindex = 0;
	int endindex = 0;
	int noiseLength, startRow;
	// noiseLength = (int)srcImage.cols/50;//抗干扰边界长
	// startRow = (int)srcImage.rows / 50;//抗干扰起始检测行数
	noiseLength = 10;
	startRow = 5;
	bool inblock = false; // 是否遍历到字符位置

	for (int i = startRow; i < painty.rows; i++)
	{

		if (!inblock && (pointcount[i] >= noiseLength)) // 进入有字符区域
		{
			inblock = true;
			startindex = i;
		}
		if (inblock && (pointcount[i] <= noiseLength)) // 进入空白区
		{
			endindex = i;
			inblock = false;
			Mat roi = Mat(srcImage, Rect(noiseLength + 1, startindex, (int)srcImage.cols - noiseLength * 2 - 1, endindex - startindex + 1));
			result.push_back(roi);
		}
	}

	delete[] pointcount;
	return result;
}

Mat drawHist(Mat &srcImage)
{
	const int nimages = 1;
	int channels[] = {0};
	MatND hist;
	int dims = 1;
	int histSize[] = {256};
	float hranges[] = {0.0, 255.0};
	const float *ranges[] = {hranges};
	calcHist(&srcImage, nimages, channels, Mat(), hist, dims, histSize, ranges);
	double minValue = 0;
	double maxValue = 0;
	minMaxLoc(hist, &minValue, &maxValue);
	Mat dstImage(histSize[0], histSize[0], CV_8U, Scalar(0));
	int hpt = saturate_cast<int>(0.9 * histSize[0]);
	for (int i = 0; i < 256; i++)
	{
		float binValue = hist.at<float>(i);
		// 统计数值的缩放，增强直方图的可视化
		int realValue = saturate_cast<int>(binValue * hpt / maxValue);
		// 在256*256的黑色底板上画矩形
		rectangle(dstImage, Point(i, histSize[0] - 1), Point(i + 1, histSize[0] - realValue), Scalar(255));
	}
	return dstImage;
}
// 获得匹配相似度
double getTempMatchMaxVal(const Mat &srcImage, const Mat &templateImage)
{
	Mat result;

	// int result_cols = srcImage.cols - templateImage.cols + 1;
	// int result_rows = srcImage.rows - templateImage.rows + 1;
	// result.create(result_cols, result_rows, srcImage.type());
	result.create(srcImage.cols, srcImage.rows, srcImage.type());

	// enum { TM_SQDIFF=0, TM_SQDIFF_NORMED=1, TM_CCORR=2, TM_CCORR_NORMED=3, TM_CCOEFF=4, TM_CCOEFF_NORMED=5 };
	matchTemplate(srcImage, templateImage, result, TM_CCOEFF_NORMED);

	double minVal = -1, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	return maxVal;
}

void trimImage(Mat &srcImage)
{
	int *h = getHorPrjArr(srcImage);
	int upLoc = 0, downLoc = (int)srcImage.rows - 1;
	int blankFlag = 0;
	for (int i = 0; i < (int)srcImage.rows; i++)
	{
		if (!blankFlag)
		{
			if (h[i] != 0)
			{
				upLoc = i;
				blankFlag = 1;
			}
		}
		else
		{
			if (h[i] == 0)
			{
				downLoc = i;
				break;
			}
		}
	}
	/*int* v = getVerPrjArr(srcImage);
	int leftLoc = 0, rightLoc = (int)srcImage.cols - 1;
	blankFlag = 0;
	for (int j = 0; j < (int)srcImage.cols; j++)
	{
		if (!blankFlag)
		{
			if (v[j] != 0)
			{
				leftLoc = j;
				blankFlag = 1;
			}
		}
		else
		{
			if (v[j] == 0)
			{
				rightLoc = j;
				break;
			}
		}
	}*/
	srcImage = Mat(srcImage, Rect(0, upLoc, (int)srcImage.cols, downLoc - upLoc + 1));
	// srcImage = Mat(srcImage, Rect(leftLoc, upLoc, rightLoc-LeftLoc+1, downLoc - upLoc + 1));
	delete[] h;
	// delete[] v;
}

vector<Mat> processImage(Mat &srcImage)
{ // 调整图片大小
	// int limSize = 800;//图片尺寸阈值
	// if (srcImage.cols > limSize || srcImage.rows > limSize)
	//	resize(srcImage, srcImage, Size(srcImage.cols /3*2, srcImage.rows /3*2));
	// cout << srcImage.cols << " , " << srcImage.rows << endl;
	// cout << 650 << " , " << (int)srcImage.rows * 650.0 / srcImage.cols << endl;
	int specificWidth = 700;
	resize(srcImage, srcImage, Size(specificWidth, (int)(srcImage.rows * ((double)specificWidth / srcImage.cols))));

	Mat grayImage, processedImage;

	// 灰度化
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	// imshow("灰度图", grayImage);

	// 滤波
	Mat medianImage;
	medianBlur(grayImage, medianImage, 3);
	// imshow("滤波后", grayImage);

	// 直方图
	Mat histImage = drawHist(medianImage);
	// imshow("hist", histImage);

	// 二值化
	// int nostuthreshold = Otsu(median);//确定阈值
	// threshold(median, result, nostuthreshold, 255, THRESH_BINARY);
	// int blockSize = ((medianImage.cols / 25) * (medianImage.rows / 25)) | 1;//计算区块的面积
	int blockSize = 25;
	int offSet = 11; // 差值
	adaptiveThreshold(medianImage, processedImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, blockSize, offSet);
	// ADAPTIVE_THRESH_GAUSSIAN_C ADAPTIVE_THRESH_MEAN_C
	// imshow("threshold", processedImage);

	// 漫水填充条形码
	int theRow = (int)processedImage.rows / 2;
	for (int i = 0; i < (int)processedImage.cols; i++)
		if (processedImage.at<uchar>(theRow, i) == 0)
			floodFill(processedImage, Point(i, theRow), Scalar(255));

	// 水平投影
	/*Mat hor = getHorPrjMat(processedImage);
	imshow("horPrj", hor);
	moveWindow("horPrj", 100, 200);*/

	// 水平分割
	vector<Mat> horPrjMat = splitImageHor(processedImage);

	// 选择有效的横向投影
	int selectedIndex = 0;
	int specificHeight = 20;
	for (int i = 0; i < horPrjMat.size(); i++)
	{
		/*cout << horPrjMat[i].rows << endl;
		imshow("hor"+to_string(i), horPrjMat[i]);
		moveWindow("hor"+to_string(i), 100, 200);*/
		if (horPrjMat[i].rows > specificHeight)
		{
			selectedIndex = i;
			break;
		}
	}
	imshow("horcut", horPrjMat[selectedIndex]);

	// cout <<"selectedIndex:"<< selectedIndex << endl;

	// 垂直分割字符
	vector<Mat> spiltedText = splitImageVer(horPrjMat[selectedIndex]);

	// 修剪与舍取
	long avgOfSquares = 0; // 平均面积
	for (int i = 0; i < (int)spiltedText.size(); i++)
	{
		trimImage(spiltedText[i]);
		avgOfSquares += (long)spiltedText[i].rows * (long)spiltedText[i].cols;
	}
	avgOfSquares /= (long)spiltedText.size();
	long noiseLimSize = avgOfSquares / 3; // 噪点面积阈值
	for (int i = 0; i < (int)spiltedText.size(); i++)
	{
		if (((long)spiltedText[i].rows * (long)spiltedText[i].cols) < noiseLimSize)
		{
			spiltedText.erase(spiltedText.begin() + i);
			i--;
		}
	}
	if ((int)spiltedText.size() > 4)
		spiltedText.erase(spiltedText.begin(), spiltedText.begin() + 4);

	return spiltedText;
}
// 数字识别
string recogNumSeq(vector<Mat> &theTexts, vector<Mat> &textTemps)
{
	string resSeq;
	double limVal = 0.35, limValOfBar = 0.4; // 准确率阈值

	for (int i = 0; i < (int)theTexts.size(); i++)
	{ // 将每个分割字符图片与模板匹配，取相似度最高的
		double maxSimiVal = limVal;
		int theIndx = -1;
		// cout << i << ":\n";
		for (int j = 0; j < (int)textTemps.size(); j++)
		{ // 调整模板大小
			double tmpVal = 0.0;
			if (j < (int)textTemps.size() - 1)
			{
				// Mat resizedTemp = Mat(textTemps[j].size(), textTemps[j].type());
				// resize(textTemps[j], resizedTemp, Size(theTexts[i].cols, theTexts[i].rows));
				// tmpVal = getTempMatchMaxVal(theTexts[i], resizedTemp);

				Mat resizedText = Mat(theTexts[i].size(), theTexts[i].type());
				resize(theTexts[i], resizedText, Size(textTemps[j].cols, textTemps[j].rows));
				tmpVal = getTempMatchMaxVal(textTemps[j], resizedText);

				if (tmpVal >= maxSimiVal)
				{
					maxSimiVal = tmpVal;
					theIndx = j;
				}
			}
			else
			{
				resize(theTexts[i], theTexts[i], Size(textTemps[j].cols, (int)((double)theTexts[i].rows * ((double)textTemps[j].cols / theTexts[i].cols))));
				tmpVal = getTempMatchMaxVal(textTemps[j], theTexts[i]);

				if (tmpVal >= maxSimiVal && tmpVal >= limValOfBar)
				{
					maxSimiVal = tmpVal;
					theIndx = j;
				}
			}
			// cout<<"  "<< j << " " << tmpVal << endl;
		}
		// cout << "##" << theIndx << " " << maxSimiVal << endl;
		if (theIndx < ((int)textTemps.size() - 1) && theIndx != -1)
		{
			if (theIndx < 10)
				resSeq.push_back('0' + theIndx);
			else
				resSeq.push_back('X');
		}
		/*if (theIndx == -1)
			resSeq.push_back('*');*/
	}
	return resSeq;
}

// 废用代码
// namespace abd
//{
//	Mat changeContrastAndLight(const Mat& srcImage, double alpha, double beta)
//	{
//		Mat dstImage;
//		dstImage.create(srcImage.size(), srcImage.type());
//		for (int x = 0; x < (int)srcImage.rows; x++)
//		{
//			for (int y = 0; y < (int)srcImage.cols; y++)
//			{
//				dstImage.at<uchar>(x, y) = saturate_cast<uchar>(alpha * (srcImage.at<uchar>(x, y)) + beta);
//			}
//		}
//		return dstImage;
//	}
//
//	vector<Mat> textOrientation(const Mat srcImage)
//	{
//		int* h = getHorPrjArr(srcImage);
//		int blankFlag = 0, topLoc = 0, bottomLoc = 0;
//		for (int x = 0; x < (int)srcImage.rows; x++)
//		{
//			if (!blankFlag)
//			{
//				if (h[x])
//				{
//					topLoc = x;
//					blankFlag = 1;
//				}
//			}
//			else
//			{
//				if (!h[x])
//				{
//					bottomLoc = x;
//					break;
//				}
//			}
//		}
//		Mat partImage(srcImage, Rect(0, topLoc, (int)srcImage.cols - 1, bottomLoc - topLoc + 1));
//
//		imshow("part_image", partImage);
//		int* v = getVerPrjArr(partImage);
//		int leftLoc = 0, rightLoc = 0;
//		blankFlag = 0;
//		vector<Mat> spiltText;
//		for (int y = 0; y < (int)partImage.cols; y++)
//		{
//			if (!blankFlag)
//			{
//				if (v[y])
//				{
//					leftLoc = y;
//					blankFlag = 1;
//				}
//			}
//			else
//			{
//				if (!v[y])
//				{
//					rightLoc = y;
//					blankFlag = 0;
//					spiltText.push_back(Mat(partImage, Rect(leftLoc, 0, rightLoc - leftLoc + 1, (int)partImage.rows - 1)));
//				}
//			}
//		}
//		for (int i = 0; i < (int)spiltText.size(); i++)
//			imshow(to_string(i), spiltText[i]);
//		return spiltText;
//	}
//
//	void drawTemplateMatchingRegion(const Mat& srcImage, const Mat& templateImage)
//	{
//		Mat result;
//		int result_cols = srcImage.cols - templateImage.cols + 1;
//		int result_rows = srcImage.rows - templateImage.rows + 1;
//		result.create(result_cols, result_rows, CV_8UC1);
//		//enum { TM_SQDIFF=0, TM_SQDIFF_NORMED=1, TM_CCORR=2, TM_CCORR_NORMED=3, TM_CCOEFF=4, TM_CCOEFF_NORMED=5 };
//		matchTemplate(srcImage, templateImage, result, TM_CCOEFF_NORMED);
//		double minVal = -1, maxVal;
//		Point minLoc, maxLoc, matchLoc, mathLoc2;
//		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
//
//		//取大值(视匹配方法而定)
//		matchLoc = maxLoc;
//		mathLoc2 = minLoc;
//
//		Mat mask = srcImage.clone();
//		rectangle(mask, matchLoc, Point(matchLoc.x + templateImage.cols, matchLoc.y + templateImage.rows), Scalar(0, 255, 0), 2, 8, 0);
//		imshow("mask", mask);
//	}
//
//	long getImageSum(Mat& srcImage)
//	{
//		long sum = 0;
//		for (int i = 0; i < (int)srcImage.cols; i++)
//			for (int j = 0; j < (int)srcImage.rows; j++)
//				sum += srcImage.at<uchar>(j, i);
//		return sum;
//	}
//
//	bool cmpRectLoc(Rect& rcA, Rect& rcB)
//	{
//		return rcA.x < rcB.x;
//	}
//
//	void recogROI(Mat& image)
//	{
//		//截取下半部分；满水填充，填条形码
//		Mat srcImage(image, Rect(0, (int)image.rows / 4 * 3, (int)image.cols, (int)image.rows / 4));
//		imshow("cut", srcImage);
//
//		//反色
//		bitwise_not(srcImage, srcImage);
//
//		//轮廓检测
//		Mat conImage = Mat::zeros(srcImage.size(), srcImage.type());
//		vector< vector<Point> > contours;
//		vector<Vec4i> hierarchy;
//		findContours(srcImage, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);
//		drawContours(conImage, contours, -1, 255);
//		cout << (int)contours.size() << endl;
//		imshow("drawContours", conImage);
//
//		//轮廓按坐标排序
//		vector<Rect> contourRect;
//		for (int i = 0; i < (int)contours.size(); i++)
//			contourRect.push_back(boundingRect(contours[i]));
//		sort(contourRect.begin(), contourRect.end(), cmpRectLoc);
//
//		//加载模板
//		vector<Mat> numTemplate;
//		Mat tmpTemplate = imread("template\\9.png");
//		cvtColor(tmpTemplate, tmpTemplate, COLOR_BGR2GRAY);
//		numTemplate.push_back(tmpTemplate);
//		imshow("temp_9", numTemplate[0]);
//
//		//轮廓尺寸归一
//		vector<Mat> ROI;
//		for (int i = 0; i < (int)contourRect.size(); i++)
//		{
//			Mat tmpROI;
//			tmpROI = conImage(contourRect[i]);
//			Mat dstROI = Mat::zeros(numTemplate[0].size(), numTemplate[0].type());
//			resize(tmpROI, dstROI, numTemplate[0].size(), 0, 0, INTER_NEAREST);
//			ROI.push_back(dstROI);
//		}
//
//		//模板匹配
//		//bitwise_not(srcImage, srcImage);
//		//templateMatching(srcImage,numTemplate[0]);
//		vector<int> seq;
//		for (int i = 0; i < (int)ROI.size(); i++)
//		{
//			if (i < 9)
//			{
//				floodFill(ROI[i], Point(50, 80), Scalar(255));
//				bitwise_not(ROI[i], ROI[i]);
//				Mat subImage;
//				absdiff(ROI[i], numTemplate[0], subImage);
//				imshow(to_string(i), subImage);
//				moveWindow(to_string(i), (i + 1) * 70, 100);
//			}
//
//			int sum = 0, min_seq = 0, mmin = 1000;
//			Mat subImage;
//			absdiff(ROI[i], numTemplate[0], subImage);
//			sum = getImageSum(subImage);
//			if (sum < mmin)
//			{
//				mmin = sum;
//				min_seq = 9;
//				seq.push_back(9);
//			}
//			sum = 0;
//		}
//
//		//输出序列
//		for (int i = 0; i < (int)seq.size(); i++)
//			cout << seq[i] << " ";
//	}
// }