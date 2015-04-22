/*#include "NLayer.h"
#include "ONLayer.h"
#include "FNLayer.h"
#include "INLayer.h"
#include "LNLayer.h"
#include "CNLayer.h"
#include "PNLayer.h"
#include "LayerGroup.h"
#include "VLayerGroup.h"
#include "ALayerGroup.h"
#include "CPVLayerGroup.h"
//#include "HLayerGroup.h"*/

#include "LayerRegister.h"

#include "Random.h"
#include <iostream>
#include <sstream>

using namespace std;


NLayer::NLayer(std::string aLayerID):
        fLayerID(aLayerID),fInitialized(false),
        fnNeuron(0),fnInput(0),fnOutput(0),fnWeight(0),
        fPrevLayerRange{0,0}, fNextLayerRange{0,0},
        fConvoPars{0,0},
        fLayerType(kUninitializedLayer),
        fInputBias(0.),
        fnData(0),
        fActivationType(kUninitializedActivation),
        fCostFunctionType(kUninitializedCost),
        fLayerData(NULL),
        fPrevLayer(NULL),fNextLayer(NULL),
        fMomentumAlpha(0.),fGlobalLearningRate(2.0),fL2Reg(0.),
        fLayer(0),fnLayer(0)

{
 static counter_t instances=0;
 fInstanceCount=instances++;
}

NLayer::~NLayer()
{

}

std::ostream& operator<<(std::ostream& aOStream, NLayer* aL)
{
  aOStream<<aL->fLayerType<<" ";
  aOStream<<aL->fLayerID<<" ";
  aOStream<<aL->fnNeuron<<" ";
  aOStream<<aL->fnInput<<" ";
  aOStream<<aL->fnOutput<<" ";
  aOStream<<aL->fnWeight<<" ";
  aOStream<<aL->fPrevLayerRange<<" ";
  aOStream<<aL->fNextLayerRange<<" ";
  aOStream<<aL->fConvoPars<<" ";
  aOStream<<aL->fInputBias<<" ";
  aOStream<<aL->fActivationType<<" ";
  aOStream<<aL->fCostFunctionType<<" ";
  aOStream<<aL->fnLayer<<endl;
  if(aL->fnLayer>0){//this is a layer group!
    if(aL->fnNeuron>0)aOStream<<aL->fLayerData;
    for(counter_t ilayer=0;ilayer<aL->fnLayer;ilayer++){
      aOStream<<aL->fLayer[ilayer]->get_layer_type()<<endl;
      aOStream<<aL->fLayer[ilayer];
    }
  }
  else{
    aOStream<<aL->fLayerData;
  }

  return aOStream;
}

std::istream& operator>>(std::istream& aIStream, NLayer* aL)
{
  if(aL->fLayerType==kUninitializedLayer)ERROR("Cannot load uninitialized layer");
  aIStream>>aL->fLayerType;
  aIStream>>aL->fLayerID;
  aIStream>>aL->fnNeuron;
  aIStream>>aL->fnInput;
  aIStream>>aL->fnOutput;
  aIStream>>aL->fnWeight;
  aIStream>>aL->fPrevLayerRange;
  aIStream>>aL->fNextLayerRange;
  aIStream>>aL->fConvoPars;
  aIStream>>aL->fInputBias;
  aIStream>>aL->fActivationType;
  aIStream>>aL->fCostFunctionType;
  aIStream>>aL->fnLayer;

  if(aL->fnLayer==0){//a simple layer
    aL->initialize_layer_data();
    aIStream>>aL->fLayerData;
  }
  else{//a layer group
    aL->fLayer.resize(aL->fnLayer);
    if(aL->fnNeuron>0){
      aL->initialize_layer_data();
      aIStream>>aL->fLayerData;
    }
    LayerType type;
    for(counter_t ilayer=0;ilayer<aL->fnLayer;ilayer++){
      aIStream>>type;
      aL->fLayer[ilayer]=LayerRegister::BuildNewLayer(type);
      if(aL->fLayer[ilayer]==NULL){
        WARN("Trying to load null layer");
        continue;
      }
      aIStream>>aL->fLayer[ilayer];
    }
    aL->initialize();
  }

  return aIStream;
}

void NLayer::print(counter_t aDataIndex){
  initialize();

  cout<<"  "<<fLayerID.c_str()<<" ("<<get_layer_type_str()<<") with "
      <<fnNeuron<<" neurons, "<<fnInput<<" inputs, "<<fnOutput<<" outputs "
      <<fnWeight<<" weights per neuron"<<endl;
  if(fnLayer!=0)cout<<"  Sub-layers in this layer: "<<fnLayer<<endl;
  if(fActivationType!=kUninitializedActivation)cout<<"  Activation: "<<get_activation_type_str()<<endl;
  if(fCostFunctionType!=kUninitializedCost)cout<<"  Cost Function: "<<get_cost_function_type_str()<<endl;
  if(fPrevLayer!=NULL){
    cout<<"  Prev layer: "<<fPrevLayer->fLayerID<<" ("<<fPrevLayer->get_layer_type_str()<<") "
        <<"range: "<<fPrevLayerRange
        <<endl;
  }
  if(fNextLayer!=NULL){
    cout<<"  Next layer: "<<fNextLayer->fLayerID<<" ("<<fNextLayer->get_layer_type_str()<<") "
        <<"range: "<<fNextLayerRange
        <<endl;
  }

  //print out layer data if this is not a layer group:
  if(fLayerData!=NULL && fnLayer==0)fLayerData->print(aDataIndex);
}

std::string NLayer::get_new_layer_id()
{
  std::ostringstream ss;
  ss<<fInstanceCount;
  return get_layer_type_str()+"_"+ss.str();
}

std::string NLayer::get_layer_type_str(LayerType aType){
  if(aType==kUninitializedLayer)return LayerRegister::LayerType2String(fLayerType);
  else return LayerRegister::LayerType2String(aType);
}
