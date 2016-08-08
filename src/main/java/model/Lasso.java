package model;

import Utils.LabeledData;
import math.DenseVector;
import java.util.List;

/**
 * Created by 王羚宇 on 2016/8/7.
 */
public class Lasso {
    public double lassoLoss(List<LabeledData> list, DenseVector model_x, DenseVector model_z, double lambda) {
        double loss = 0.0;
        for (LabeledData labeledData: list) {
            double predictValue = model_x.dot(labeledData.data);
            loss += 1.0 / 2.0 * Math.pow(labeledData.label - predictValue, 2);
        }
        for(Double v: model_z.values){
            loss += lambda * (v > 0? v : -v);
        }
        return loss;
    }
    public double test(List<LabeledData> list, DenseVector model) {
        double residual = 0;
        for (LabeledData labeledData : list) {
            double dot_prod = model.dot(labeledData.data);
            residual += 0.5 * Math.pow(labeledData.label - dot_prod, 2);
        }
        return residual;
    }
    public double lassoLoss(List<LabeledData> list, DenseVector model, double lambda) {
        double loss = 0.0;
        for (LabeledData labeledData: list) {
            double predictValue = model.dot(labeledData.data);
            loss += 1.0 / 2.0 * Math.pow(labeledData.label - predictValue, 2);
        }
        for(Double v: model.values){
            loss += lambda * (v > 0? v : -v);
        }
        return loss;
    }
}
