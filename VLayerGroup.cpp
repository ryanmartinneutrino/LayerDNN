#include "VLayerGroup.h"
#include <iostream>

using namespace std;

VLayerGroup::VLayerGroup(std::string aLayerID):
            LayerGroup(aLayerID)
{
  fLayerType=kVerticalLayerGroup;
}

VLayerGroup::~VLayerGroup()
{
  //dtor
}

void VLayerGroup::initialize(bool aForce)
{
  if(fInitialized && !aForce)return;
  if(fLayerID=="noid")fLayerID=get_new_layer_id();
  if(fLayer.size()==0)ERROR("No layers");
  if(fLayer[0]==NULL)ERROR("First layer is null!");

  //First and last layer pointers for easy access to the input and output of the stack:
  fFirstLayer=fLayer[0];
  fLastLayer=fLayer[fnLayer-1];

  fnOutput=fLastLayer->get_noutput();
  fnInput=fFirstLayer->get_ninput();

  //if some of the sub-layers are layer groups, now is the time to initialize them
  for(auto layer: fLayer)layer->initialize();

  initialize_pointers();
  initialize_layer_data();
  fInitialized=true;
}

void VLayerGroup::initialize_pointers()
{
  //initialize the prev and next layer pointers in the layers in the stack
  //also set the the layers to span the full range of prev and next
  for(counter_t ilayer=0;ilayer<fnLayer-1;ilayer++){
    fLayer[ilayer]->set_next_layer_ptr(fLayer[ilayer+1]);
    fLayer[ilayer+1]->set_prev_layer_ptr(fLayer[ilayer]);

    fLayer[ilayer]->set_next_layer_range(0,fLayer[ilayer+1]->get_nneuron()-1);
    fLayer[ilayer+1]->set_prev_layer_range(0,fLayer[ilayer]->get_noutput()-1);
  }

}

void VLayerGroup::initialize_layer_data()
{
  if(fLayerData==NULL){
    fLayerData=fLastLayer->get_layer_data_ptr();
  }
}

void VLayerGroup::forward_pass(const TR2 &aX, counter_t aStart, counter_t anData)
{ //Forward pass in a vertical layer group assumes that layer 0 is the only one to take the data
  //check for errors

  initialize();
  if(fLayer.size()==0)ERROR("No layer");
  if(fLayer[0]==NULL)ERROR("First layer is null!");

  //if data was passed, then only give it to the first layer!
  if(aX.size()!=0) fLayer[0]->forward_pass(aX,aStart,anData);
  else fLayer[0]->forward_pass();

  for(counter_t ilayer=1;ilayer<fnLayer;ilayer++){
    fLayer[ilayer]->forward_pass();
  }

}

void VLayerGroup::backprop_pass(const TR2 &aY, counter_t aStart, counter_t anData)
{
  if(fLayer.size()==0)ERROR("No layers for backprop pass");

  //initialize pointers between layers in the group
  initialize_pointers();
  if(aY.size()!=0){
    fLayer[fnLayer-1]->backprop_pass(aY,aStart,anData);
  }
  else fLayer[fnLayer-1]->backprop_pass();
  if(fnLayer<2)return;
  for(int ilayer=int(fnLayer-2);ilayer>-1;ilayer--){
    fLayer[ilayer]->backprop_pass();
  }
}

void VLayerGroup::copy_targets(const TR2 &aY, counter_t aStart, counter_t anData)
{
  if(aY.size()!=0 /*&& fLayer[fnLayer-1]->get_layer_type()==kOutputLayer*/){
    fLayer[fnLayer-1]->backprop_pass(aY,aStart,anData);
  }
  else ERROR("empty data");
}

