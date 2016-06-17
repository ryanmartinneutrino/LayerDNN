
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


int main(int argc, char* argv[]){

  counter_t ninput=0;
  counter_t noutput=0;
  TR3 trainingLN(0);
  TR3 validation(0);
  LoadIrisDataLN("iris.dat",trainingLN,ninput,noutput,148);

  //input layer
  INLayer ilayer(ninput);

  //Fullly connected layers  
  FNLayer flayer1(ilayer.get_noutput(),8,1.0,"FC1");
  flayer1.set_activation_type(kLogisticActivation);
  flayer1.set_global_learning_rate(0.03);  
    
  FNLayer flayer2(flayer1.get_noutput(),5,1.0,"FC2");
  flayer2.set_activation_type(kLogisticActivation);
  flayer2.set_global_learning_rate(0.03);
    
  //Output layer
  ONLayer olayer(flayer2.get_noutput(),noutput,1.0,"output");
  olayer.set_cost_function_type(kQuadraticCost);
  olayer.set_activation_type(kSoftMaxActivation);

  VLayerGroup vlg("Master");
  vlg.add_layer(&ilayer);
  vlg.add_layer(&flayer1);
  vlg.add_layer(&flayer2);
  vlg.add_layer(&olayer);
  vlg.print();

  LayerTrainer Trainer2(&vlg);
  Trainer2.set_classif_threshold(0.9);
  Trainer2.set_target_classif_rate(0.99);  
 // Trainer2.set_target_cost(0.01);
  Trainer2.set_nmax_epochs(10000);
  Trainer2.train_mini_batches(trainingLN,trainingLN,4);


/**/
  return 0;
}
