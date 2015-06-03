#include "LayerData.h"
#include "Random.h"
#include "NLayer.h"

#include <math.h>
#include <iostream>


LayerData::LayerData(NLayer* aLayer):
           fLayer(aLayer),fInputLayerData(NULL),
           fW(0),fAvgDeltaW(0),fAvgDeltaWPrev(0),fLearningRateGain(0),
           fOutput(0),fZ(0),fDelta(0),fDeltaW(0),fOutputFactor(0)

{

}

LayerData::~LayerData()
{
 delete fInputLayerData;
}

//!! I/O related methods and streams:

std::ostream& operator<<(std::ostream& aOStream, LayerData* aLD)
{//only writes out the weights
  counter_t n1=aLD->fW.size();
  if(n1==0)return aOStream;
  counter_t n2=aLD->fW[0].size();
  if(n2==0)return aOStream;

  for(counter_t i1=0;i1<n1;i1++){//loop over neurons
    for(counter_t i2=0;i2<n2;i2++){//loop over weights
      aOStream<<aLD->fW[i1][i2]<<" ";
    }
    aOStream<<endl;
  }
  return aOStream;
}

std::istream& operator>>(std::istream& aIStream, LayerData* aLD)
{//Only reads in the weights, assumes an NLayer has already initialized the dimensions of the arrays

  counter_t n1=aLD->fW.size();
  if(n1==0)return aIStream;
  counter_t n2=aLD->fW[0].size();
  if(n2==0)return aIStream;

  for(counter_t i1=0;i1<n1;i1++){//loop over neurons
    for(counter_t i2=0;i2<n2;i2++){//loop over weights
      aIStream>>aLD->fW[i1][i2];
    }
  }
  return aIStream;
}


void LayerData::print(counter_t aDataIndex)
{
  if(fLayer==NULL)return;

  cout<<"  Layer Data:"<<endl;
  counter_t nNeuron=fW.size();
  //Print weights if there are more than 0 neurons:
  if(nNeuron>0){
    for(counter_t ineuron=0;ineuron<nNeuron;ineuron++){
      cout<<"    Neuron:"<<ineuron<<" weights :";
      counter_t nInput=fW[ineuron].size();
      for(counter_t in=0;in<nInput;in++){
        cout<<fW[ineuron][in]<<" ";
      }
     cout<<endl;
    }
  }
  counter_t nData=fOutput.size();
  if(nData>aDataIndex){
    counter_t nOutput=fOutput[aDataIndex].size();
    if(nOutput>0){
      cout<<"    Outputs in this Layer: ";
      for(counter_t io=0;io<nOutput;io++)cout<<fOutput[aDataIndex][io]<<" ";
      cout<<endl;
    }
  }
  //else WARN("Data index may be out of range!");
  cout<<endl;

}

//!! Initialization related methods:

void LayerData::initialize_weights(bool aRandomize, number_t aFixedW)
{ //Initialize the weight matrices for a layer with neurons and inputs
  counter_t nNeuron=fLayer->get_nneuron();
  counter_t nWeight=fLayer->get_nweight();
  if(nNeuron==0)return;
  if(nWeight==0)return;

  fW.resize(nNeuron);
  fAvgDeltaW.resize(nNeuron);
  fAvgDeltaWPrev.resize(nNeuron);
  fLearningRateGain.resize(nNeuron);
  number_t sqrtn=1./sqrt(number_t(nWeight));
  for(counter_t ineuron=0;ineuron<nNeuron;ineuron++){
    fW[ineuron].resize(nWeight,aFixedW);
    fLearningRateGain[ineuron].resize(nWeight,1.0);
    fAvgDeltaW[ineuron].resize(nWeight,0.);
    fAvgDeltaWPrev[ineuron].resize(nWeight,0.);
    if(aRandomize){
      for(counter_t in=0;in<nWeight;in++){
        fW[ineuron][in]=Random::GetRandomGaussian(0.,sqrtn);
      }
    }
  }
}

void LayerData::initialize_ndata(const counter_t anData, bool aForce)
{
  if(fLayer==NULL) ERROR("NULL point to fLayer");
  //cout<<fLayer->get_layer_type_str().c_str()<<" ndata = "<<anData<<endl;
  if(anData==0)return;
  if(fLayer->get_ndata()==anData && !aForce){return;}//already initialized

  fLayer->set_ndata(anData);

  counter_t nOutput=fLayer->get_noutput();

  if(nOutput!=0){
    fOutput.resize(anData);
    fOutputFactor.resize(anData);
    for(counter_t idata=0;idata<anData;idata++){
      fOutput[idata].resize(nOutput,0.);
      fOutputFactor[idata].resize(nOutput,1.0);//used for max pooling
    }
  }
  //Only layers with neurons can have Z, W, deltaW
  counter_t nNeuron=fLayer->get_nneuron();
  if(nNeuron!=0){
    fZ.resize(anData);
    fDeltaW.resize(anData);
    fDelta.resize(anData);
    counter_t nWeight=fLayer->get_nweight();
    for(counter_t idata=0;idata<anData;idata++){
      fDelta[idata].resize(nNeuron,0.);
      fZ[idata].resize(nNeuron,0.);
      fDeltaW[idata].resize(nNeuron);
      for(counter_t ineuron=0;ineuron<nNeuron;ineuron++){
        fDeltaW[idata][ineuron].resize(nWeight,0.);
      }
    }
  }
}


//!! Feed forward related methods:

void LayerData::copy_data_to_output(const TR2 &aX, counter_t aStart, counter_t anData)
{
  counter_t ndata=aX.size();
  if(ndata==0)return;//no copy if no data
  if(anData!=0){//check that range is not exceeded
    ndata=(aStart+anData>ndata? ndata-aStart : anData);
  }

  initialize_ndata(ndata);//this will call the inherited version if appropriate!

  counter_t ndataOut=aX[0].size();
  //check that output is initialized to be big enough (if it has a bias, it can be bigger than fn2!)
  if(fOutput.size()!=ndata || fOutput[0].size()<ndataOut)ERROR("fOutput not initialized");

  //The actual copy operation
  for(counter_t idata=0;idata<ndata;idata++){//loop over data set
    for(counter_t io=0;io<ndataOut;io++){//loop over inputs in data
      fOutput[idata][io]=aX[aStart+idata][io];
    }
  }
}

void LayerData::copy_data_to_output(const std::vector<LayerData*> &aInputLayerData)
{
  counter_t nLayer=aInputLayerData.size();
  if(nLayer<1)ERROR("No input layer data");
  if(aInputLayerData[0]==NULL)ERROR("NULL first layer data");
  counter_t nData=aInputLayerData[0]->fOutput.size();
  if(nData==0)ERROR("No data");

  counter_t nLayerOutputs;
  counter_t layerOffset=0;

  initialize_ndata(nData);

  for(counter_t ilayer=0;ilayer<nLayer;ilayer++){
    nLayerOutputs=aInputLayerData[ilayer]->fOutput[0].size();
    //cout<<"at layer "<<ilayer<<" "<<nLayerOutputs<<" "<<fOu
    for(counter_t idata=0;idata<nData;idata++){
       for(counter_t io=0;io<nLayerOutputs;io++){
         fOutput[idata][layerOffset+io]=aInputLayerData[ilayer]->fOutput[idata][io];
       }
    }
    layerOffset+=nLayerOutputs-1;
  }


}

void LayerData::feed_forward(const TR2 &aX, counter_t aStart, counter_t anData, ActivationType aType)
{//creates a dummy layerdata to copy the data to and pass as input for the normal feed forward method.
  counter_t ndata=aX.size();
  if(ndata==0)return;//no copy if no data
  if(anData!=0){//check that range is not exceeded
    ndata=(aStart+anData>ndata? ndata-aStart : anData);
  }

  initialize_ndata(ndata);//this will call the inherited version if appropriate!
  counter_t ndataOut=aX[0].size();
  if(fInputLayerData==NULL)fInputLayerData=new LayerData();

  fInputLayerData->fOutput.resize(ndata);
  for(counter_t idata=0;idata<ndata;idata++){//loop over data set
      fInputLayerData->fOutput[idata].resize(ndataOut);
    for(counter_t io=0;io<ndataOut;io++){//loop over inputs in data
      fInputLayerData->fOutput[idata][io]=aX[aStart+idata][io];
    }
  }
  feed_forward(*fInputLayerData,aType);
}


void LayerData::feed_forward(const LayerData &aPrevLayer, ActivationType aType)
{  //!! could have more checks.
  counter_t nData=aPrevLayer.fOutput.size();//number of data sets
  counter_t nNeuron=fW.size();//number of neurons in this layer

  if(nNeuron<1 || nData<1){
   WARN("Trying to feed forward with no data or no neurons")
   return;
  }
  initialize_ndata(nData);

  //The range of inputs to this layer may not span the full range
  //of output of the next layer...
  counter_t nPrevFirst=fLayer->get_prev_layer_first();
  counter_t nWeight=fLayer->get_nweight();
  number_t inputBias=fLayer->get_input_bias();
  if(inputBias>0.)nWeight--;//decrease by one and handle bias separately

  //If this is a local receptive field (such as a convolution layer) need to make sure
  //we don't exceed the input range when offsetting by the convolution delta:
  counter_t nConvoDelta=fLayer->get_convo_delta();
  if(aPrevLayer.fOutput[0].size()<nPrevFirst+nConvoDelta*(nNeuron-1)+nWeight){
    cout<<fLayer->get_layer_type_str().c_str()<<endl;
    cout<<aPrevLayer.fOutput[0].size()<<" "<<nWeight<<endl;
    ERROR("Prev layer output too short with this convo delta");
  }
  bool doConvo=false;
  if(fLayer->get_layer_type()==kConvolutionLayer)doConvo=true;

  number_t norm=0.;
  counter_t prevOffset=nPrevFirst;

  for(counter_t idata=0;idata<nData;idata++){//loop over data set
    if(aType==kSoftMaxActivation)norm=0.;
    for(counter_t ineuron=0;ineuron<nNeuron;ineuron++){//loop over neuron in layers
      if(nConvoDelta!=0)prevOffset=ineuron*nConvoDelta+nPrevFirst;
      fZ[idata][ineuron]=0;
      for(counter_t iprev=0;iprev<nWeight;iprev++){
        fZ[idata][ineuron]+=(doConvo ? fW[0][iprev] :fW[ineuron][iprev] )
                            *aPrevLayer.fOutput[idata][prevOffset+iprev];
      }
      if(inputBias>0.)fZ[idata][ineuron]+=fW[ineuron][nWeight];//bias term!
      switch(aType){
        case(kLogisticActivation):
          fOutput[idata][ineuron]=1./(1.+exp(-fZ[idata][ineuron]));//logistic
          break;
        case(kTanhActivation):
          fOutput[idata][ineuron]=(exp(fZ[idata][ineuron])-exp(-fZ[idata][ineuron]))
                                  /(exp(fZ[idata][ineuron])+exp(-fZ[idata][ineuron]));//logistic
          break;
        case(kReLUActivation):
          fOutput[idata][ineuron]=( fZ[idata][ineuron]>0 ? fZ[idata][ineuron] : 0 );//Rectified Linear Unit
          break;
        case(kSoftMaxActivation):
          fOutput[idata][ineuron]=exp(fZ[idata][ineuron]);//softmax
          norm+=fOutput[idata][ineuron];
          break;
        default:
          ERROR("Activation Function not set");
      }//end switch
     }//end of loop over neurons
     //SoftMax normalization:
     if(aType==kSoftMaxActivation){
      if(norm==0)ERROR("Normalization of zero in softmax!");
      for(counter_t ineuron=0;ineuron<nNeuron;ineuron++){
        fOutput[idata][ineuron]/=norm;
      }
     }//end normalization for softmax
    }//end of loop over data
}

void LayerData::feed_forward_pool(LayerData &aPrevLayer)
{  //!! could have more checks.
  counter_t nData=aPrevLayer.fOutput.size();//number of data sets
  counter_t nNeuron=fW.size();//number of neurons in this layer

  if(nNeuron<1 || nData<1){
   WARN("Trying to feed forward with no data or no neurons")
   return;
  }
  initialize_ndata(nData);
  //The range of inputs to this layer may not span the full range
  //of output of the next layer...
  counter_t nPrevFirst=fLayer->get_prev_layer_first();
  counter_t nWeight=fLayer->get_nweight();
  //use convo delta for a pooling layer
  counter_t nConvoDelta=fLayer->get_convo_delta();
  if(aPrevLayer.fOutput[0].size()<nPrevFirst+nConvoDelta*(nNeuron-1)+nWeight){
    ERROR("Prev layer output too short with this convo delta");
  }

  counter_t prevOffset=nPrevFirst;
  counter_t imax=0;
  for(counter_t idata=0;idata<nData;idata++){//loop over data set
    for(counter_t ineuron=0;ineuron<nNeuron;ineuron++){//loop over neuron in layers
      if(nConvoDelta!=0)prevOffset=ineuron*nConvoDelta+nPrevFirst;
      fZ[idata][ineuron]=0;
      //max pooling:
      for(counter_t iprev=0;iprev<nWeight;iprev++){
        if(aPrevLayer.fOutput[idata][prevOffset+iprev]>fZ[idata][ineuron]){
          fZ[idata][ineuron]=aPrevLayer.fOutput[idata][prevOffset+iprev];
          imax=iprev;
        }
      }
      fOutput[idata][ineuron]=fZ[idata][ineuron];//max pooling
      for(counter_t iprev=0;iprev<nWeight;iprev++){
       if(iprev==imax)aPrevLayer.fOutputFactor[idata][prevOffset+iprev]=1.;
       else aPrevLayer.fOutputFactor[idata][prevOffset+iprev]=0.;
      }
     }//end of loop over neurons
    }//end of loop over data
}

//!! Backprop related methods
void LayerData::just_delta_from_next_layer(const LayerData &aNextLayer)
{//only calculate delta, not DeltaW (for aggregating layer)
  counter_t nData=aNextLayer.fOutput.size();
  counter_t nNeuron=fW.size();
  if(nNeuron<1)ERROR("No neurons in layer");
  if(nData<1)ERROR("No data");

  counter_t nNextFirst=fLayer->get_next_layer_first();
  counter_t nNextLast=fLayer->get_next_layer_last();
  counter_t nNextNeuron=nNextLast-nNextFirst+1;
 // counter_t nNextNeuron=aNextLayer.fW.size();
  counter_t nextOffset=nNextFirst;


  for(counter_t idata=0;idata<nData;idata++){//loop over data set
    for(counter_t ineuron=0;ineuron<nNeuron;ineuron++){//loop over neuron in layers
      fDelta[idata][ineuron]=0.;
        //Calculate delta from the next layer
        for(counter_t inextn=0;inextn<nNextNeuron;inextn++){
          //WT*d (transpose of weight matrix with delta):
          fDelta[idata][ineuron]+=aNextLayer.fW[inextn+nextOffset][ineuron]
                                  *aNextLayer.fDelta[idata][inextn+nextOffset]
                                  *fOutputFactor[idata][ineuron];
        }

    }//end loop over neurons
  }//end loop over the data

}

void LayerData::delta_from_next_layer(const LayerData &aPrevLayer, const LayerData &aNextLayer,
                                      ActivationType aType)
{

  counter_t nData=aNextLayer.fOutput.size();
  counter_t nNeuron=fW.size();
  if(nNeuron<1)ERROR("No neurons in layer");
  if(nData<1)ERROR("No data");
  if(fLayer==nullptr)ERROR("Null Layer pointer");

  //indices of relevant neurons in the next layer
  counter_t nNextFirst=fLayer->get_next_layer_first();
  counter_t nNextLast=fLayer->get_next_layer_last();
  counter_t nNextNeuron=nNextLast-nNextFirst+1;

  if(nNextLast>aNextLayer.fW.size())ERROR("Out of range in next layer!");

  //indices of relevant neurons in the previous layer
  counter_t nPrevFirst=fLayer->get_prev_layer_first();
  counter_t nWeight=fLayer->get_nweight();
  number_t inputBias=fLayer->get_input_bias();
  if(inputBias>0.)nWeight--;//decrease by 1 and handle bias separately

  //If this is a local receptive field (such as a convolution layer) need to make sure
  //we don't exceed the input range:
  counter_t nConvoDelta=fLayer->get_convo_delta();
  if(aPrevLayer.fOutput[0].size()<nPrevFirst+nConvoDelta*(nNeuron-1)+nWeight){
    ERROR("Output too short with this convo delta");
  }

  //if the next layer is a receptive field of this one, need to modify the
  //range for the WT calculation:
  counter_t nConvoDeltaNext=0;
  counter_t nOverlapNext=0;
  counter_t nSpanNext=0;
  if(fLayer->get_next_layer_ptr()!=NULL){
    nConvoDeltaNext=fLayer->get_next_layer_ptr()->get_convo_delta();
    nOverlapNext=fLayer->get_next_layer_ptr()->get_convo_overlap();
    nSpanNext=fLayer->get_next_layer_ptr()->get_convo_span();
  }
  if(nConvoDeltaNext != nSpanNext-nOverlapNext)ERROR("delta not set correctly for span and overlap");

  counter_t prevOffset=nPrevFirst;
  counter_t nextOffset=nNextFirst;
  counter_t nNextRelevantNeurons=nNextNeuron;
/*
  cout<<"in "<<fLayer->get_layer_type_str().c_str()<<"  "<<prevOffset<<" "
      <<nNextNeuron<<" "<<aNextLayer.fOutput[0].size()<<" "<<nConvoDeltaNext<<" "
      <<nPrevFirst<<" "<<fDeltaW[0][0].size()<<" "
      <<endl;*/

  for(counter_t idata=0;idata<nData;idata++){//loop over data set

    for(counter_t ineuron=0;ineuron<nNeuron;ineuron++){//loop over neuron in layers
      fDelta[idata][ineuron]=0.;
      //Calculate WT * delta from the next layer to get this layer:
      //if Z=0 and ReLU, can skip, since delta will be zero:
      //cout<<"delta for neuron "<<ineuron<<" in "<<fLayer->get_layer_type_str().c_str()
      //    <<" "<<nConvoDeltaNext<<" "<<idata<<endl;
      if( !(aType==kReLUActivation && fZ[idata][ineuron]<=0.) ){
        //if next layer is a local receptive field (e.g. convolution),
        //need to change the indices around:
        if(nConvoDeltaNext!=0){//!!only works if overlap==0
          //cout<<"here for "<<fLayer->get_layer_type_str().c_str()<<" "<<nConvoDeltaNext<<endl;
          nNextRelevantNeurons=1;//only true for no overlap!!
          counter_t start=ineuron/nConvoDeltaNext;
          nextOffset=nNextFirst+start;

        }

        if(aNextLayer.fW.size()<nextOffset+nNextRelevantNeurons){
          ERROR("Next weight matrix too small!");
        }

        //Now, actually calculate delta from the next layer
        for(counter_t inextn=0;inextn<nNextRelevantNeurons;inextn++){
          //WT*d (transpose of weight matrix with delta):
          fDelta[idata][ineuron]+=aNextLayer.fW[inextn+nextOffset][ineuron]
                                  *aNextLayer.fDelta[idata][inextn+nextOffset]
                                  *fOutputFactor[idata][ineuron];//0 or 1 if this goes into a pooling layer, otherwise 1
        }
      }

      //Multiply delta by derivative of activation function
      //Derivative of ReLU is 1, no need to multiply delta by 1
      if(aType==kLogisticActivation || aType==kSoftMaxActivation){
        fDelta[idata][ineuron]*=fOutput[idata][ineuron]*(1.-fOutput[idata][ineuron]);//logistic and softmax derivative
      }
      if(aType==kTanhActivation){
        fDelta[idata][ineuron]*=(1.-fOutput[idata][ineuron]*fOutput[idata][ineuron]);//derivative of tanh
      }

      //Update the delta W and sum that into the AvgDeltaW.
      //Note that update_weights will divide by nData:
      for(counter_t in=0;in<nWeight;in++){
        //cout<<nWeight<<" "<<fDeltaW[idata][ineuron].size()<<" "<<aPrevLayer.fOutput[idata].size()<<endl;
        fDeltaW[idata][ineuron][in]=fDelta[idata][ineuron]*aPrevLayer.fOutput[idata][prevOffset+in];
        if(idata==0)fAvgDeltaW[ineuron][in]=fDeltaW[idata][ineuron][in];//instead of initializing to zero
        else fAvgDeltaW[ineuron][in]+=fDeltaW[idata][ineuron][in];//this is the sum, needs to be averaged before weight update
      }
      //Handle bias separately (this is why we had weights--)
      if(inputBias>0.){
        fDeltaW[idata][ineuron][nWeight]=fDelta[idata][ineuron];
        if(idata==0)fAvgDeltaW[ineuron][nWeight]=fDeltaW[idata][ineuron][nWeight];//instead of initializing to zero
        else fAvgDeltaW[ineuron][nWeight]+=fDeltaW[idata][ineuron][nWeight];//this is the sum, needs to be averaged before weight update
      }

    }//end loop over neurons
  }//end loop over the data
}


void LayerData::update_weights_avgDeltaW()
{
  counter_t nNeuron=fW.size();//fn1
  if(nNeuron<1){
    WARN("trying to update weights with no neurons")
    return;
  }
  counter_t nInput=fW[0].size();//fn2
  if(nInput<1){
    WARN("trying to update with no inputs")
    return;
  }
  counter_t nData=fDelta.size();
  if(nData==0){
    WARN("trying to update weights with no data")
    return;
  }
  number_t dndata=number_t(nData);
  number_t momentumAlpha=fLayer->get_momentum_alpha();
  number_t globalLearningRate=fLayer->get_global_learning_rate();
  number_t learningRate=globalLearningRate;
  number_t L2Reg=fLayer->get_L2_regularization();

  bool doConvo=false;
  if(fLayer->get_layer_type()==kConvolutionLayer)doConvo=true;


  for(counter_t ineuron=0;ineuron<nNeuron;ineuron++){//loop over neurons
    for(counter_t in=0;in<nInput;in++){

      //!!Adaptive weight gains (need to make the adjustments adjustable and have a flag)
      /*
      if(fAvgDeltaW[ineuron][in]*fAvgDeltaWPrev[ineuron][in]>0){
        fLearningRateGain[ineuron][in]+=0.05;
      }
      else fLearningRateGain[ineuron][in]*=0.95;*/

      learningRate=globalLearningRate*fLearningRateGain[ineuron][in];

      fAvgDeltaW[ineuron][in]=fAvgDeltaW[ineuron][in]/dndata;
      fAvgDeltaWPrev[ineuron][in]=fAvgDeltaW[ineuron][in];
      if(!doConvo){
        fW[ineuron][in]-=learningRate*fAvgDeltaW[ineuron][in]
                         +momentumAlpha*fAvgDeltaWPrev[ineuron][in]//momentum term (to speed up learning)
                         +learningRate*L2Reg*fW[ineuron][in];//L2 regularization (weight decay to prevent overfitting)
      }
      if(doConvo && ineuron!=0)fAvgDeltaW[0][in]+=fAvgDeltaW[ineuron][in];
    }
  }
  if(doConvo){//average the deltaW and apply those to the weights of neuron 0
    for(counter_t in=0;in<nInput;in++){
      /*
      if(fAvgDeltaW[0][in]*fAvgDeltaWPrev[0][in]>0){
        fLearningRateGain[0][in]+=0.05;
      }
      else fLearningRateGain[0][in]*=0.95;*/
      learningRate=globalLearningRate*fLearningRateGain[0][in];
      fAvgDeltaW[0][in]=fAvgDeltaW[0][in]/number_t(nNeuron);
      fAvgDeltaWPrev[0][in]=fAvgDeltaW[0][in];
      fW[0][in]-=learningRate*fAvgDeltaW[0][in]
                 +momentumAlpha*fAvgDeltaWPrev[0][in]//momentum term (to speed up learning)
                 +learningRate*L2Reg*fW[0][in];//L2 regularization (weight decay to prevent overfitting)
      for(counter_t ineuron=0;ineuron<nNeuron;ineuron++){
        fW[ineuron][in]=fW[0][in];
      }
    }
  }


}











