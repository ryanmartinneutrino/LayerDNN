
#include <NLayer.h> //abstract base class
#include <INLayer.h>//input layer
#include <FNLayer.h>//fully connected layer
#include <ONLayer.h>//output layer (has targets in LayerData)
#include <LNLayer.h>//Local receptive field layer
#include <CNLayer.h>//convolution layer
#include <PNLayer.h>//pooling layer
#include <VLayerGroup.h>//vertical group of layers
#include <ALayerGroup.h>
#include <CPVLayerGroup.h>


#include <LayerTrainer.h>

#include <sstream>
#include <thread>
#include <fstream>
#include <math.h>
#include <random>
#include <iostream>

using namespace std;

void LoadWFAEDataLN(string filename, TR3 &training, counter_t &nInput, counter_t &nOutput, counter_t maxTraining=0, counter_t start=0)
{
  ifstream infile(filename);//space separated, converted from NIST CSV from Kaggle

  nInput=256;
  nOutput=nInput;
  counter_t ntraining=0;
  TR1 input(nInput);
  TR1 output(nOutput);

  counter_t out=0;
  counter_t nlines=0;
  number_t in;
  char buffer[100000];
  infile.getline(buffer,100000);

  training.resize(2);

  while(!infile.eof()){

    for(size_t i=0;i<nInput;i++){
      infile>>in;
      input[i]=in;
    }

    if(infile.eof())break;
    for(size_t i=0;i<nInput;i++){
      infile>>out;
      output[i]=out;
    }

    nlines++;
    if(nlines>=start){
      training[0].push_back(input);
      training[1].push_back(output);
      ntraining++;
    }
    if(ntraining>=maxTraining && maxTraining!= 0)break;
  }

}


void LoadWFDataLN(string filename, TR3 &training, counter_t &nInput, counter_t &nOutput, counter_t maxTraining=0, counter_t start=0)
{
  ifstream infile(filename);//space separated, converted from NIST CSV from Kaggle

  nInput=256;
  nOutput=2;
  counter_t ntraining=0;
  TR1 input(nInput);
  TR1 output(nOutput);

  counter_t out=0;
  counter_t nlines=0;
  number_t in;
  char buffer[100000];
  infile.getline(buffer,100000);

  training.resize(2);

  while(!infile.eof()){

    for(size_t i=0;i<nInput;i++){
      infile>>in;
      input[i]=in;
    }

    if(infile.eof())break;
    infile>>out;
    output.assign(nOutput,0.0);//make the targets slightly away from 1 and 0 for a logistic
    output[out]=1.0;

    nlines++;
    if(nlines>=start){
      training[0].push_back(input);
      training[1].push_back(output);
      ntraining++;
    }
    if(ntraining>=maxTraining && maxTraining!= 0)break;
  }

}

void LoadFakeDataLN(string filename, TR3 &training, counter_t &nInput,counter_t &nOutput, counter_t maxTraining=0, counter_t start=0)
{
  ifstream infile(filename);//space separated, converted from NIST CSV from Kaggle

  nInput=128;
  nOutput=5;
  counter_t ntraining=0;
  TR1 input(nInput);
  TR1 output(nOutput);

  counter_t out=0;
  counter_t nlines=0;
  number_t in;
  char buffer[100000];
  infile.getline(buffer,100000);

  training.resize(2);
  while(!infile.eof()){

    if(infile.eof())break;
    for(size_t i=0;i<nInput;i++){
      infile>>in;
      input[i]=in;
    }

    infile>>out;

    output.assign(nOutput,0.0);
    output[out]=1.;
    nlines++;
    if(nlines>=start){
      training[0].push_back(input);
      training[1].push_back(output);
      ntraining++;
    }

    if(ntraining>=maxTraining && maxTraining!= 0)break;
  }

}

void LoadMNISTDataLN(string filename, TR3 &training, counter_t &nInput,counter_t &nOutput, counter_t maxTraining=0, counter_t start=0)
{
  ifstream infile(filename);//space separated, converted from NIST CSV from Kaggle

  nInput=784;
  nOutput=10;
  counter_t ntraining=0;
  TR1 input(nInput);
  TR1 output(nOutput);

  counter_t out=0;
  counter_t nlines=0;
  number_t in;
  char buffer[100000];
  infile.getline(buffer,100000);

  training.resize(2);
  while(!infile.eof()){
    infile>>out;
    if(infile.eof())break;
    output.assign(nOutput,0.0);
    output[out]=1.;

    for(size_t i=0;i<nInput;i++){
      infile>>in;
      input[i]=in;
    }

    nlines++;
    if(nlines>=start){
      training[0].push_back(input);
      training[1].push_back(output);
      ntraining++;
    }
    if(ntraining>=maxTraining && maxTraining!= 0)break;
  }

}

void LoadIrisDataLN(string filename, TR3 &training, counter_t &nInput, counter_t &nOutput, counter_t maxTraining=0, counter_t start=0)
{
  ifstream infile(filename);//space separated, converted from NIST CSV from Kaggle

  nInput=4;
  nOutput=3;
  counter_t ntraining=0;
  TR1 input(nInput);
  TR1 output(nOutput);

  counter_t out=0;
  number_t in;
  char buffer[100000];
  infile.getline(buffer,100000);

  training.resize(2);

  while(!infile.eof()){

    for(size_t i=0;i<nInput;i++){
      infile>>in;
      input[i]=in;
    }

    if(infile.eof())break;
    infile>>out;
    output.assign(nOutput,0.0);//make the targets slightly away from 1 and 0 for a logistic
    output[out]=1.0;

    if(ntraining>=start){
      training[0].push_back(input);
      training[1].push_back(output);
      ntraining++;
    }
    if(ntraining>=maxTraining && maxTraining!= 0)break;
  }

}

//!!TODO
//Does momentum work?
//Does LearningRateGain in LayerData work?
//Does the L2 regularization work?

int main(int argc, char* argv[]){

  counter_t ninput=0;
  counter_t noutput=0;
  TR3 trainingLN(0);
  TR3 validation(0);
  //LoadIrisDataLN("iris.dat",trainingLN,ninput,noutput,148);
  LoadFakeDataLN("fakedata.dat",trainingLN,ninput,noutput,1000);
  //LoadFakeDataLN("fakedata.dat",validation,ninput,noutput,1000,1000);

  //LoadWFAEDataLN("TrainingWaveforms_112.dat",trainingLN,ninput,noutput,1000);
  //LoadWFDataLN("TrainingWaveforms_112.dat",validation,ninput,noutput,1000,1000);
  TR2 testX,testY;//training
  TR2 valX,valY;//validation
  for(int i=0;i<5;i++){
    testX.push_back(trainingLN[0][i]);
    testY.push_back(trainingLN[1][i]);
    valX.push_back(trainingLN[0][i]);
    valY.push_back(trainingLN[1][i]);
  }

  //input layer
  INLayer ilayer(ninput);

/*
  //autoencoder:
  FNLayer flayer1(ilayer.get_noutput(),128,1.0,"FC1");
  flayer1.set_activation_type(kTanhActivation);
  FNLayer flayer2(flayer1.get_noutput(),64,1.0,"FC2");
  flayer2.set_activation_type(kTanhActivation);


  ONLayer olayer(flayer2.get_noutput(),noutput,1.0,"output");
  olayer.set_cost_function_type(kQuadraticCost);
  olayer.set_activation_type(kTanhActivation);

  VLayerGroup vlg("Master");
  vlg.add_layer(&ilayer);
  vlg.add_layer(&flayer1);
  vlg.add_layer(&flayer2);
  vlg.add_layer(&olayer);
  flayer2.print();

  LayerTrainer Trainer2(&vlg);
  //Trainer2.set_classif_threshold(0.9);
  Trainer2.set_target_cost(0.01);
  Trainer2.set_nmax_epochs(20);
  Trainer2.train_mini_batches(trainingLN,trainingLN,5);*/

  //convo net
  //*

  //Local receptive field
  counter_t nSpan=4,nOverlap=0;
  LNLayer llayer(nSpan,nOverlap,ilayer.get_noutput(),1.0,"LRF1");

  //fully connected layers
  FNLayer flayer(llayer.get_noutput(),7,1.0,"FC1");
  flayer.set_activation_type(kReLUActivation);
  flayer.set_global_learning_rate(0.02);
  flayer.set_momentum_alpha(0.1);

  VLayerGroup vlgG01("VLG1:LRF1_FC1");
  vlgG01.add_layer(&llayer);
  vlgG01.add_layer(&flayer);

  //Convo layer
  CNLayer clayer(nSpan,nOverlap,ilayer.get_noutput(),"CN1");
  PNLayer player(4,clayer.get_noutput(),"PL1");

  FNLayer flayer2(player.get_noutput(),7,1.0,"FC2");
  flayer2.set_activation_type(kTanhActivation);
  flayer2.set_global_learning_rate(0.02);
  flayer2.set_momentum_alpha(0.1);
  FNLayer flayer3=flayer2;
  FNLayer flayer4=flayer2;

  CPVLayerGroup vlgG02(nSpan,4,ilayer.get_noutput());
  vlgG02.add_layer(&flayer2);

  CPVLayerGroup vlgG03(nSpan,4,ilayer.get_noutput());
  vlgG03.add_layer(&flayer3);

  CPVLayerGroup vlgG04(nSpan,4,ilayer.get_noutput());
  vlgG04.add_layer(&flayer4);

  //aggregate the fully connected layers
  ALayerGroup alg("ALG:VLG1_FC2");
  alg.add_layer(&vlgG01);
  alg.add_layer(&vlgG02);
  alg.add_layer(&vlgG03);
  //alg.add_layer(&vlgG04);

  ONLayer olayer(alg.get_noutput(),noutput,1.0,"output");
  olayer.set_cost_function_type(kXCorrelCost);
  olayer.set_activation_type(kSoftMaxActivation);

  VLayerGroup vlg("Master");
  vlg.add_layer(&ilayer);
  vlg.add_layer(&alg);
  vlg.add_layer(&olayer);
  vlg.print();

  LayerTrainer Trainer2(&vlg);
  Trainer2.set_classif_threshold(0.9);
  Trainer2.set_target_classif_rate(0.998);
  Trainer2.set_target_cost(0.02);
  Trainer2.set_nmax_epochs(300);
  Trainer2.train_mini_batches(trainingLN,trainingLN,50);

  ofstream ofile("vlg_trained.dat");
  ofile<<vlg;
  ofile.close();

  cout<<"Loading from file:"<<endl;
  ifstream ifile("vlg_trained.dat");
  VLayerGroup vgfile;
  ifile>>vgfile;
  vgfile.forward_pass(testX);
  vgfile.get_last_layer()->print();
/**/
  return 0;
}
