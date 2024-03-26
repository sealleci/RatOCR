#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "simpleORC.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace std;
using namespace cv;

void loadTemplates(vector<Mat> &);
string splitFileName(string);
void loadFiles(vector<cv::String> &, string);
int LCS(string &, string &);
void evaluateAccuracy(vector<int> &, vector<cv::String> &, vector<Mat> &);
void testCode();

int main()
{
    // testCode();

    vector<Mat> charTemplates;
    cout << ">加载模板中" << endl;
    loadTemplates(charTemplates); // 加载模板
    cout << ">模板加载完成" << endl;

    vector<cv::String> fileNames;
    string directory = "completion_set"; // 文件夹名
    cout << ">读取文件中" << endl;
    loadFiles(fileNames, directory); // 遍历文件夹下所有图片,fileNames储存文件名
    cout << ">文件读取完成" << endl;

    vector<int> resData;
    cout << ">开始识别与匹配" << endl;
    evaluateAccuracy(resData, fileNames, charTemplates); // resData来存识别与匹配的数据
    cout << ">识别与匹配完成" << endl;

    cout << ">读取文件个数：" << resData[0] << ",识别正确个数：" << resData[1] << endl;
    cout << ">正确率：" << fixed << setprecision(2) << (double)resData[1] / (double)resData[0] * 100 << "%" << endl;
    cout << ">总字符数：" << resData[2] << ",识别正确的字符数：" << resData[3] << endl;
    cout << ">准确率：" << fixed << setprecision(2) << (double)resData[3] / (double)resData[2] * 100 << "%" << endl;

    return 0;
}

void loadTemplates(vector<Mat> &charTemp)
{
    for (int i = 0; i < 10; i++)
    {
        Mat tmpChar = imread("template\\" + to_string(i) + ".png");
        cvtColor(tmpChar, tmpChar, COLOR_BGR2GRAY);
        bitwise_not(tmpChar, tmpChar);
        charTemp.push_back(tmpChar);
    }
    Mat tmpChar = imread("template\\X.png");
    cvtColor(tmpChar, tmpChar, COLOR_BGR2GRAY);
    bitwise_not(tmpChar, tmpChar);
    charTemp.push_back(tmpChar);
    tmpChar = imread("template\\-.png");
    cvtColor(tmpChar, tmpChar, COLOR_BGR2GRAY);
    bitwise_not(tmpChar, tmpChar);
    charTemp.push_back(tmpChar);
}

string splitFileName(string fileName) // 把文件名中需要识别的地方从中分割出来
{
    string res;
    size_t i = fileName.length() - 5; // 用int 提示可能丢失数据
    while ((fileName[i] >= 48 && fileName[i] <= 57) || fileName[i] == '-' || fileName[i] == 'X')
    {
        if ((fileName[i] >= 48 && fileName[i] <= 57) || fileName[i] == 'X')
        {
            res.insert(0, 1, fileName[i]); // 在0位前插入1个字符；
        }
        i--;
    }
    return res;
}

void loadFiles(vector<cv::String> &fileNames, string directory)
{
    cv::String namePattern = "./" + directory + "/*.jpg";
    // cout << namePattern << endl;
    glob(namePattern, fileNames, false);
}

int LCS(string &s1, string &s2)
{
    const int l1 = (int)s1.length();
    const int l2 = (int)s2.length();
    int **dp = new int *[l1 + 1];
    for (int i = 0; i < l1 + 1; i++)
        dp[i] = new int[l2 + 1]();
    for (int i = 1; i <= l1; i++)
    {
        for (int j = 1; j <= l2; j++)
        {
            if (s1[i - 1] == s2[j - 1])
                dp[i][j] = dp[i - 1][j - 1] + 1;
            else
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
        }
    }
    int res = dp[l1][l2];
    for (int i = 0; i < l1 + 1; i++)
        delete[] dp[i];
    delete[] dp;
    return res;
}

void evaluateAccuracy(vector<int> &resData, vector<cv::String> &fileNames, vector<Mat> &charTemplates)
{
    size_t cntOfFiles = fileNames.size(); // 读取文件数
    int cntOfRightSeqs = 0;               // 完全识别正确的个数
    size_t sumOfChars = 0;                // 总字符数
    int cntOfRightChars = 0;              // 识别正确字符数
    vector<string> storedFileNames, storedRecNames;
    vector<int> isMatching;
    cout << "   >识别序列：" << endl;
    for (int i = 0; i < (int)cntOfFiles; i++)
    {
        Mat curImage = imread(fileNames[i]);
        string splitedName = splitFileName(fileNames[i]); // 把文件名中需要识别的地方从中分割出来
        storedFileNames.push_back(splitedName);

        // 判断处理结果recognizedName与splitedName是否匹配
        sumOfChars += splitedName.length();

        vector<Mat> charsArr;
        string recognizedName;

        try
        {
            charsArr = processImage(curImage);
            recognizedName = recogNumSeq(charsArr, charTemplates);
            storedRecNames.push_back(recognizedName);
        }
        catch (...)
        {
            recognizedName = "error";
            storedRecNames.push_back(recognizedName);
        }

        if (splitedName == recognizedName)
        {
            cntOfRightSeqs++;
            isMatching.push_back(1);
        }
        else
        {
            isMatching.push_back(0);
        }

        if (i % 8 == 0)
        {
            cout << "   ";
        }
        cout << "[" << i + 1 << "]";
        if ((i + 1) % 8 == 0 || i == cntOfFiles - 1)
        {
            cout << endl;
        }
        else
        {
            cout << ",";
        }
    }

    cout << "   >打印结果：" << endl;
    for (int i = 0; i < (int)cntOfFiles; i++)
    {
        cout << "   #" << storedFileNames[i] << "  ~  ";
        cout << left << setw((int)storedFileNames[i].length()) << storedRecNames[i] << "  ";

        if (storedFileNames[i] == storedRecNames[i])
        {
            cout << "[FITTED]" << endl;
        }
        else
        {
            cout << "[UNFITTED]" << endl;
        }

        cntOfRightChars += LCS(storedFileNames[i], storedRecNames[i]);
    }

    resData.push_back(cntOfFiles);
    resData.push_back(cntOfRightSeqs);
    resData.push_back(sumOfChars);
    resData.push_back(cntOfRightChars);
}

void testCode()
{
    string imgName[] = {"ISBN 978-7-5068-3441-4.jpg", "ISBN 978-7-115-43978-9.jpg", "978-7-5398-4931-7.jpg", "ISBN 978-7-302-15567-6.jpg", "ISBN 978-7-111-40701-0.jpg", "ISBN 7-302-09260-5.jpg"};
    Mat devil = imread("completion_set\\" + imgName[0]);
    vector<Mat> arr = processImage(devil);
    cout << arr.size() << endl;
    for (int i = 0; i < (int)arr.size(); i++)
    {
        imshow(to_string(i), arr[i]);
        moveWindow(to_string(i), 70 * (i + 1), 100);
    }
    vector<Mat> tp;
    loadTemplates(tp);
    string res = recogNumSeq(arr, tp);
    cout << res << endl;

    waitKey(1000000);
}
