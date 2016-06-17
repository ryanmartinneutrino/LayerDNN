#include "LayerTrainer.h"
#include <random>
#include <algorithm>
#include <iostream>

using namespace std;

LayerTrainer::LayerTrainer(NLayer* aLayer):
              fLayer(aLayer),
              fnMaxEpoch(1000),
              fClassifThreshold(0.5),
              fReduceLRFactor(0.95),fIncreaseLRFactor(1.),fReduceLRTerm(0.),fIncreaseLRTerm(0.05),
              fClassifRateChange(0.01),fCostChange(0.01),
              fTargetCost(-1.0),fTargetClassifRate(-1.0)
{
  //ctor
}

LayerTrainer::~LayerTrainer()
{
  //dtor
}

void LayerTrainer::one_epoch_mini_batches(TR3 &aXY, counter_t anPerBatch)
{
  if(fLayer==NULL)ERROR("NULL fLayer");

  //!!This does not guarantee that they got shuffles in the same order on a different platform!!!
  counter_t tim=time(NULL) % rand();

  srand(tim);
  random_shuffle(aXY[0].begin(), aXY[0].end());
  srand(tim);
  random_shuffle(aXY[1].begin(), aXY[1].end());

  counter_t nperbatch=anPerBatch;
  counter_t ndata=aXY[0].size();

  if(nperbatch==0 || nperbatch>ndata)nperbatch=ndata;//if 0, full batch mode
  counter_t nbatch=ndata/nperbatch;//this rounds down, which should be correct!

  counter_t start=0;
  //cout<<"partitioning the data into "<<nbatch<<" of "<<nperbatch<<" per batch"<<endl;
  for(counter_t ibatch=0;ibatch<nbatch;ibatch++){

    start=ibatch*nperbatch;
    if(start+nperbatch>ndata)break;//not enough data left in the training set (if you include a smaller data set, it is effectively weighted more (which may be ok for random data)

    fLayer->forward_pass(aXY[0],start,nperbatch);
    fLayer->backprop_pass(aXY[1],start,nperbatch);
    //cout<<"about to update weights "<<ibatch<<" "<<nbatch<<endl;
    fLayer->update_weights();
  }
}

void LayerTrainer::train_mini_batches(TR3 &aTraining, const TR3 &aValidation,
                                      counter_t anPerBatch)
{
  cout<<"Training "<<fLayer->get_layer_type_str().c_str()
      <<"with "<<aTraining[0].size()<<" data with mini batches of "<<anPerBatch<<endl;
  cout<<"Cost function type: "<<fLayer->get_last_layer()->get_cost_function_type_str().c_str()<<endl;

  if(fTargetCost<0. && fTargetClassifRate<0.){
    cout<<"Defaulting to using targe classification rate of 0.999";
    fTargetClassifRate=0.999;
  }
  else if(fTargetCost<0){
    cout<<"Using target classification rate of "<<fTargetClassifRate<<endl;
  }
  else{
    cout<<"Using target cost of "<<fTargetCost<<endl;
  }

  number_t cost=0.;
  number_t prevCost=0.;
  number_t classificationRate=1.;
  number_t prevclassificationRate=0.;
  counter_t ntraining=0;
  counter_t ndata=aTraining[0].size();

  counter_t nweight_updates_perset=1;//number of weight updates after looping through the data once (1=full batch)
  if(anPerBatch!=0)nweight_updates_perset=ndata/anPerBatch;
  counter_t update_learning_period=100;//20/nweight_updates_perset;//weight until at least 100 weight updates before calculating cost and evaluating performance
  bool converged=false;

  for(counter_t i=0;i<fnMaxEpoch && !converged ;i++){
    
    one_epoch_mini_batches(aTraining,anPerBatch);
    
    //how often to update the learning rate
    if( (i % update_learning_period) ==0){
      
      //run on validation data
      fLayer->forward_pass(aValidation[0]);
      fLayer->copy_targets(aValidation[1]);

      //calculate metrics and adjust learning rate
      //!!need to be careful with increasing or decreasing the learning rate, it can explode!
      if(fTargetCost<0){//use classification rate maximization
        classificationRate=fLayer->get_classification_success_rate(fClassifThreshold);
        //decrease learning rate if classification got worse
        if(classificationRate<prevclassificationRate){
          fLayer->multiply_global_learning_rate(fReduceLRFactor);
        }
        //If classification rate has not changed by more than 1%
        else if(fabs(classificationRate-prevclassificationRate)/classificationRate <fClassifRateChange){
          fLayer->add_to_global_learning_rate(fIncreaseLRTerm);
        }
        else{}
        if(classificationRate>fTargetClassifRate)converged=true;
        prevclassificationRate=classificationRate;
      }
      else{//use cost minimization
        cost=fLayer->get_cost();
        if(cost>prevCost){
          fLayer->multiply_global_learning_rate(fReduceLRFactor);
        }
        //If classification rate has not changed by more than 1%
        else if(fabs(cost-prevCost)/cost <fCostChange){
          fLayer->add_to_global_learning_rate(fIncreaseLRTerm);
        }
        else{}
        if(cost<fTargetCost)converged=true;
        prevCost=cost;
      }
      
      //Print progress
      if(i % (fnMaxEpoch/20)==0){
        cout<<"Epoch "<<i<<" of "<<fnMaxEpoch<<" ";
        if(fTargetCost>-1.0){
          cout<<"cost: "<<cost<<endl;
        }
        else{
          cout<<"classif rate: "<<classificationRate<<endl;
        }
      }

      /*
      //!!Make decisions to optimize the learning or stop training (mutually exclusive optimizations)
      //decrease learning rate if classification and cost got worse
      if(classificationRate<prevclassificationRate || cost>prevCost){
        //cout<<"reducing learning rate by a factor of "<<fReduceLRFactor<<endl;
        fLayer->multiply_global_learning_rate(fReduceLRFactor);
      }
      //If cost and classification rate have not changed by more than 1%
      else if(fabs(classificationRate-prevclassificationRate)/classificationRate <fClassifRateChange &&
              fabs(cost-prevCost)/cost <fCostChange){
        //need to be careful with increasing or decreasing the learning rate, it can explode!
        //if classification rate is less than target, keep going with a bigger learning rate
       fLayer->add_to_global_learning_rate(fIncreaseLRTerm);
       //cout<<"increasing learning rate by "<<fIncreaseLRTerm<<endl;

      }
      else{}
      if(classificationRate>fTargetClassifRate)converged=true;
      prevclassificationRate=classificationRate;
      prevCost=cost; */

    }     
    ntraining++;
  }
  //One more run on the test data
  fLayer->forward_pass(aValidation[0]);
  fLayer->copy_targets(aValidation[1]);
  //calculate metrics
  cost=fLayer->get_cost();
  classificationRate=fLayer->get_classification_success_rate(fClassifThreshold);
  if(ntraining==fnMaxEpoch)cout<<"Did not meet convergence criteria!!"<<endl;
  cout<<"Terminated training after "<<ntraining<<" epochs"<<endl;
  cout<<"Final cost: "<<cost<<" classif rate: "<<classificationRate<<endl;
  cout<<"***********"<<endl;
}
