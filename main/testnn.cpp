
#include "../src/NLayer.h" //abstract base class
#include "../src/INLayer.h"//input layer
#include "../src/FNLayer.h"//fully connected layer
#include "../src/ONLayer.h"//output layer (has targets in LayerData)
#include "../src/LNLayer.h"//Local receptive field layer
#include "../src/CNLayer.h"//convolution layer
#include "../src/PNLayer.h"//pooling layer
#include "../src/VLayerGroup.h"//vertical group of layers
#include "../src/ALayerGroup.h"
#include "../src/CPVLayerGroup.h"

#include "../src/LayerTrainer.h"

#include <sstream>
#include <thread>
#include <fstream>
#include <math.h>
#include <random>
#include <iostream>

using namespace std;
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

  //LoadWFDataLN("TrainingWaveforms_112.dat",trainingLN,ninput,noutput,1000);
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
  flayer2.set_activation_type(kReLUActivation);
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

//*
  LayerTrainer Trainer2(&vlg);
  Trainer2.set_classif_threshold(0.9);
  Trainer2.set_target_classif_rate(0.998);
  Trainer2.set_nmax_epochs(100);
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
/*
  //Local receptive field
  counter_t nSpan=8,nOverlap=0;
  LNLayer llayer(nSpan,nOverlap,ilayer.get_noutput(),1.0,0,0);
  //llayer.set_activation_type(kReLUActivation);
  llayer.set_global_learning_rate(1.0);
  llayer.set_momentum_alpha(0.1);


  //Fully connected layer
  FNLayer flayer(llayer.get_noutput(),15,1.0,0,0);
  flayer.set_activation_type(kLogisticActivation);//note that ReLU typically requires smaller learning rate...
  flayer.set_global_learning_rate(1.0);
  //flayer.set_L2_regularization(0.1);
  flayer.set_momentum_alpha(0.1);

  //Fully connected layer
  FNLayer flayer2(flayer.get_noutput(),10,1.0,0,0);
  flayer2.set_activation_type(kReLUActivation);//note that ReLU typically requires smaller learning rate...
  flayer2.set_global_learning_rate(0.5);
  //flayer2.set_momentum_alpha(0.1);
  //flayer2.set_L2_regularization(0.1);

  //Output layer
  ONLayer olayer(flayer2.get_noutput(),noutput,1.0,0,0);
  olayer.set_cost_function_type(kXCorrelCost);
  olayer.set_activation_type(kSoftMaxActivation);
  olayer.set_global_learning_rate(1.0);
 // olayer.set_momentum_alpha(0.1);
  //olayer.set_L2_regularization(0.0);

  VLayerGroup vlg;
  vlg.add_layer(&ilayer);
  vlg.add_layer(&llayer);
  vlg.add_layer(&flayer);
  vlg.add_layer(&flayer2);
  vlg.add_layer(&olayer);
  vlg.initialize_layer_data();

 // vlg.print();

  LayerTrainer Trainer(&vlg);
  Trainer.set_classif_threshold(0.9);
  //Trainer.train_mini_batches(trainingLN,validation,50);




  ofstream ofile("vlg_trained.dat");
  ofile<<vlg;
  ofile.close();

  cout<<"Loading from file:"<<endl;
  ifstream ifile("vlg_trained.dat");
  VLayerGroup vgfile;
  ifile>>vgfile;
  //vgfile.print();
  vgfile.forward_pass(testX);
  //vgfile.get_last_layer()->print(0);


  /////////////Test aggregating layer
cout<<"\n\n\n --------------------------- \n\n\n";
//Try aggregating 2 groups of
  INLayer ilayerG0(ninput,0,0);
  nSpan=16,nOverlap=0;
  LNLayer llayerG0(nSpan,nOverlap,ilayerG0.get_noutput(),1.0,1,0);
  FNLayer flayerG0(llayerG0.get_noutput(),6,1.0,2,0);
  flayerG0.set_activation_type(kReLUActivation);
  flayerG0.set_global_learning_rate(0.0005);
  //flayerG0.set_L2_regularization(0.05);
  VLayerGroup vG0;
  vG0.add_layer(&ilayerG0);
  vG0.add_layer(&llayerG0);
  vG0.add_layer(&flayerG0);

  INLayer ilayerG1(ninput,0,1);
  nSpan=16,nOverlap=0;
  LNLayer llayerG1(nSpan,nOverlap,ilayerG1.get_noutput(),1.0,1,1);
  FNLayer flayerG1(llayerG1.get_noutput(),6,1.0,2,1);
  flayerG1.set_activation_type(kReLUActivation);
  flayerG1.set_global_learning_rate(0.0005);
  VLayerGroup vG1;
  vG1.add_layer(&ilayerG1);
  vG1.add_layer(&llayerG1);
  vG1.add_layer(&flayerG1);

  ALayerGroup alayer(3,0);
  alayer.add_layer(&vG0);
  alayer.add_layer(&vG1);
  alayer.initialize_layer_data();
  //alayer.forward_pass(testX);
  //alayer.print();

  ONLayer olayerG(alayer.get_noutput(),noutput,1.0,4,0);
  olayerG.set_cost_function_type(kXCorrelCost);
  olayerG.set_activation_type(kSoftMaxActivation);

  VLayerGroup superG;
  superG.add_layer(&alayer);
  superG.add_layer(&olayerG);
  superG.initialize_layer_data();
  superG.forward_pass(testX);
  superG.backprop_pass(testY);
  superG.update_weights();


  LayerTrainer Trainer2(&superG);
  Trainer2.set_classif_threshold(0.9);
  Trainer2.train_mini_batches(trainingLN,validation,50);

  superG.print();


  ofstream sgfile("superGfile.dat");
  sgfile<<superG;
  sgfile.close();

*/
  return 0;
}
