#include "OLayerData.h"
#include "ONLayer.h"

#include<iostream>
#include<math.h> //to use log()

using namespace std;

OLayerData::OLayerData(ONLayer* aLayer):
            fTarget(0)
{
  fLayer=aLayer;
}

OLayerData::~OLayerData()
{

}

void OLayerData::initialize_ndata(const counter_t anData, bool aForce)
{
  LayerData::initialize_ndata(anData,aForce);

  counter_t nOutput=fLayer->get_noutput();
  counter_t nNeuron=fLayer->get_nneuron();

  if(nNeuron!=0){
    fTarget.resize(anData);
    for(counter_t idata=0;idata<anData;idata++){
      fTarget[idata].resize(nOutput,0.);
    }
  }
}

void OLayerData::copy_data_to_target(const TR2 &aY, counter_t aStart, counter_t anData)
{
  counter_t ndata=aY.size();
  if(anData!=0){
    ndata=(aStart+anData>ndata? ndata-aStart : anData);
  }

  initialize_ndata(ndata);//should not need this

  counter_t nOutput=aY[0].size();
  for(counter_t idata=0;idata<ndata;idata++){//loop over data set
    for(counter_t io=0;io<nOutput;io++){//loop over inputs in data
      fTarget[idata][io]=aY[aStart+idata][io];
    }
  }
}

void OLayerData::delta_from_target(const LayerData &aPrevLayer, ActivationType aType,
                                  CostFunctionType aCostType)
{//!! could use  more checks!
  counter_t nData=fTarget.size();//fn1
  if(nData<1)ERROR("Target vector has no data!");
  if(fOutput.size()<nData)ERROR("Output has less data than target");

  counter_t nNeuron=fDelta[0].size();//fn2
  if(nNeuron<1){
    WARN("No neurons!");//should it just be an error?
    return;
  }

 // counter_t nInput=aPrevLayer.fOutput[0].size();
  //if(fDeltaW[0][0].size()!=nInput)ERROR("Input size mismatch!");

  counter_t nWeight=fLayer->get_nweight();//!! new (was nInput)
  number_t inputBias=fLayer->get_input_bias();
  if(inputBias>0.)nWeight--;//decrease by 1 and handle bias separately


  for(counter_t idata=0;idata<nData;idata++){//loop over data set
    for(counter_t ineuron=0;ineuron<nNeuron;ineuron++){//loop over neuron in layers
      //delta = cost derivative * activation derivative

      //delta=0 for ReLU activation if fZ<=0
      if(aType==kReLUActivation && fZ[idata][ineuron]<=0) fDelta[idata][ineuron]=0.;
      else{//cost derivative (order matters, since it changes the sign!)
        fDelta[idata][ineuron]=(fOutput[idata][ineuron]-fTarget[idata][ineuron]) ;
        if(aCostType==kQuadraticCost){
        //only need to multiply by derivative if Quadratic cost (i.e. not for cross correlation cost)
          if(aType==kLogisticActivation || aType==kSoftMaxActivation){
            //logistic and softmax derivative are the same, y*(1-y):
            fDelta[idata][ineuron]*=fOutput[idata][ineuron]*(1.-fOutput[idata][ineuron]);
          }
        }
      }
      //update the delta W and the AvgDeltaW as well:
      for(counter_t in=0;in<nWeight;in++){
        fDeltaW[idata][ineuron][in]=fDelta[idata][ineuron]*aPrevLayer.fOutput[idata][in];
        if(idata==0)fAvgDeltaW[ineuron][in]=fDeltaW[idata][ineuron][in];
        else fAvgDeltaW[ineuron][in]+=fDeltaW[idata][ineuron][in];//this is the sum, needs to be averaged before weight update
      }
      //bias part:
      if(inputBias>0.){
        fDeltaW[idata][ineuron][nWeight]=fDelta[idata][ineuron];
        if(idata==0)fAvgDeltaW[ineuron][nWeight]=fDeltaW[idata][ineuron][nWeight];//instead of initializing to zero
        else fAvgDeltaW[ineuron][nWeight]+=fDeltaW[idata][ineuron][nWeight];//this is the sum, needs to be averaged before weight update
      }
    }
  }
}

number_t OLayerData::get_cost(CostFunctionType aCostType)
{
  counter_t nData=fTarget.size();//fn1
  if(nData<1)ERROR("Target vector has no data!");
  if(fOutput.size()<nData)ERROR("Output has less data than target");
  counter_t nNeuron=fOutput[0].size();//fn2

  number_t cost=0.,d=0.;
  for(counter_t idata=0;idata<nData;idata++){//loop over data set
    for(counter_t ineuron=0;ineuron<nNeuron;ineuron++){
      if(aCostType==kQuadraticCost){
        d=fOutput[idata][ineuron]-fTarget[idata][ineuron];
        d*=d;
      }
      else if(aCostType==kXCorrelCost){
        d=fTarget[idata][ineuron]*fOutput[idata][ineuron];
        if(d<=0.)continue;
        d=-log(d);
      }
      else ERROR("Cost function not known");
      if(d!=d)continue;
      cost+=d;
    }
  }

  if(cost!=cost)ERROR("NaN cost!");
  if(aCostType==kQuadraticCost) return 0.5*cost/number_t(nData);
  if(aCostType==kXCorrelCost) return cost/number_t(nData);

  ERROR("Cost function not known");
  return -1;//should never get here!
}

number_t OLayerData::get_classification_success_rate(number_t aThreshold)
{
  counter_t nData=fTarget.size();//fn1
  if(nData<1)ERROR("Target vector has no data!");
  if(fOutput.size()<nData)ERROR("Output has less data than target");
  counter_t nNeuron=fOutput[0].size();//fn2

  number_t correct=0.;
  number_t maxtarget=-1.;
  number_t maxoutput=-1;
  counter_t itarget=0;
  counter_t ioutput=0;
  for(counter_t idata=0;idata<nData;idata++){//loop over data set
    maxtarget=-1.;
    maxoutput=-1;
    itarget=0;
    ioutput=0;
    for(counter_t ineuron=0;ineuron<nNeuron;ineuron++){//find the most active neuron in the target and the layer output
      if(fOutput[idata][ineuron]>maxoutput && fOutput[idata][ineuron]>=aThreshold){
        ioutput=ineuron;
        maxoutput=fOutput[idata][ineuron];
      }
      if(fTarget[idata][ineuron]>maxtarget){
        itarget=ineuron;
        maxtarget=fTarget[idata][ineuron];
      }
    }
    if(itarget==ioutput)correct++;
  }
  return correct/number_t(nData);
}
