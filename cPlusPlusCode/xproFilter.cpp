#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>

using namespace cv;
using namespace std;

//interpolation
void interpolation(uchar* lut, float* curve, float* originalValue){
    for(int i=0; i<256; i++){
        int j=0;
        float a = i;
        while (a>originalValue[j]){
            j++;
        }
        if(a == originalValue[j]){
            lut[i] = curve[j];
            continue;
        }
        float slope = ((float)(curve[j] - curve[j-1]))/((float)(originalValue[j] - originalValue[j-1]));
        float constant = curve[j] - slope * originalValue[j];
        lut[i] = slope * a + constant;
    }
}

int main(){

    //Read input image
    Mat image = imread("../assets/anish.jpg");

    //create a copy of the input image to work on
    Mat output = image.clone();
    cout<<output.size()<<endl;

    //apply the Vignette for Halo effect
    //define vignette scale
    float vignetteScale = 6;
    //calculate the kernel size 
    int k = std::min(output.rows, output.cols)/vignetteScale;

    //create kernel to get Halo effects
    Mat kernelX = getGaussianKernel(output.cols, k);
    Mat kernelY = getGaussianKernel(output.rows, k);
    Mat kernelx_transpose;
    transpose(kernelX, kernelx_transpose);
    Mat kernel = kernelY * kernelx_transpose;
    cout<<kernel.size()<<endl;

    //Normalize the kernel
    Mat mask;
    normalize(kernel, mask, 0, 1, NORM_MINMAX);

    //convert to float32
    output.convertTo(output, CV_32F);
    mask.convertTo(mask, CV_32F);

    //split the channels
    vector<Mat> channels;
    split(output, channels);

    //Apply halo effect to all three channels
    channels[0] = channels[0] + channels[0].mul(mask);
    channels[1] = channels[1] + channels[1].mul(mask);
    channels[2] = channels[2] + channels[2].mul(mask);

    //merge the channels
    merge(channels, output);

    output = output / 2;

    //limit the pixel value between 0 and 255
    min(output, 255, output);
    max(output, 0, output);

    //convert back to cv_8uc3
    output.convertTo(output, CV_8UC3);

    //Splitting the channels
    split(output, channels);

    //Interpolation values
    float redValuesOriginal[] = {0, 42, 105, 148, 185, 255};
    float redValues[] =         {0, 28, 100, 165, 215, 255 };
    float greenValuesOriginal[] = {0, 40, 85, 125, 165, 212, 255};
    float greenValues[] =         {0, 25, 75, 135, 185, 230, 255 };
    float blueValuesOriginal[] = {0, 40, 82, 125, 170, 225, 255 };
    float blueValues[] =         {0, 38, 90, 125, 160, 210, 222};

    //create a lookuptable
    Mat lookupTable(1, 256, CV_8U);
    uchar* lut = lookupTable.ptr();

    //apply interpolation and create lookup table for red channel
    interpolation(lut, redValues, redValuesOriginal);
    //Apply mapping for red Channel
    LUT(channels[2], lookupTable, channels[2]);

    //apply interpolation and create lookup table for green channel
    interpolation(lut, greenValues, greenValuesOriginal);
    //Apply mapping for green Channel
    LUT(channels[1], lookupTable, channels[1]);

    //apply interpolation and create lookup table for blue channel
    interpolation(lut, blueValues, blueValuesOriginal);
    //Apply mapping for blue Channel
    LUT(channels[0], lookupTable, channels[0]);

    //merge back the channels
    merge(channels, output);

    //adjust contrast
    //convert to YcrCb color space
    cvtColor(output, output, COLOR_BGR2YCrCb);

    //convert to float32
    output.convertTo(output, CV_32F);

    //split the channels
    split(output, channels);

    //Scale the Y channel
    channels[0] = channels[0] * 1.2;

    //limit the pixel values between 0 and 255
    min(channels[0], 255, channels[0]);
    max(channels[0], 0, channels[0]);

    //merge back the channels
    merge(channels, output);

    //convert back to cv_8uc3
    output.convertTo(output, CV_8UC3);

    //convert back to BGR color space
    cvtColor(output, output, COLOR_YCrCb2BGR);

    //create windows to display images
    namedWindow("image", WINDOW_AUTOSIZE);
    namedWindow("xpro", WINDOW_AUTOSIZE);

    //display images
    imshow("image", image);
    imshow("xpro", output);

    //Press esc to exit the windows
    waitKey(0);

    //close all the opened windows
    destroyAllWindows();

    return 0;
}