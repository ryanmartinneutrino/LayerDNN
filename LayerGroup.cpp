#include "LayerGroup.h"
#include <iostream>

using namespace std;

LayerGroup::LayerGroup(std::string aLayerID):
            NLayer(aLayerID),
            fFirstLayer(NULL),fLastLayer(NULL)
{
  fLayerType=kUninitializedLayerGroup;
}

LayerGroup::~LayerGroup()
{

}

void LayerGroup::print(counter_t aDataIndex)
{
  initialize();
  NLayer::print();//prints the information about the layer
  for(counter_t ilayer=0;ilayer<fnLayer;ilayer++){
    cout<<"-->sub-layer "<<ilayer<<":"<<endl;
    fLayer[ilayer]->print(aDataIndex);
  }
}

void LayerGroup::update_weights()
{
  for(counter_t ilayer=0;ilayer<fnLayer;ilayer++){
    fLayer[ilayer]->update_weights();
  }
}

void LayerGroup::multiply_global_learning_rate(number_t aFact)
{
  for(counter_t ilayer=0;ilayer<fnLayer;ilayer++){
    fLayer[ilayer]->multiply_global_learning_rate(aFact);
  }
}
void LayerGroup::add_to_global_learning_rate(number_t a)
{
  for(counter_t ilayer=0;ilayer<fnLayer;ilayer++){
    fLayer[ilayer]->add_to_global_learning_rate(a);
  }
}


