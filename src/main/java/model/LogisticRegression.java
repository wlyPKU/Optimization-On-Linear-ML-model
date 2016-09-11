package model;

import Utils.*;
import it.unimi.dsi.fastutil.doubles.DoubleComparator;
import math.DenseVector;
import java.util.List;

/**
 * Created by 王羚宇 on 2016/8/7.
 */
public class LogisticRegression {

    public void testAndSummary(List<LabeledData>trainCorpus, List<LabeledData> testCorpus,
                                DenseVector model, double lambda){
        long startTest = System.currentTimeMillis();
        double trainLoss = logLoss(trainCorpus, model, lambda);
        double testLoss = logLoss(testCorpus, model, lambda);
        double trainAuc = auc(trainCorpus, model);
        double testAuc = auc(testCorpus, model);
        long testTime = System.currentTimeMillis() - startTest;
        System.out.println("trainLoss=" + trainLoss + " testLoss=" + testLoss +
                " trainAuc=" + trainAuc + " testAuc=" + testAuc +
                " testTime=" + testTime);
    }

    public double logLoss(List<LabeledData> list, DenseVector model, double lambda) {
        double loss = 0.0;
        for (LabeledData labeledData: list) {
            double p = model.dot(labeledData.data);
            double z = p * labeledData.label;
            loss += Math.log(1 + Math.exp(-z));
        }
        for(Double v : model.values){
            loss += lambda * (v > 0? v : -v);
        }
        return loss;
    }
    public double test(List<LabeledData> list, DenseVector model) {
        int N_RIGHT = 0;
        int N_TOTAL = 0;
        for (LabeledData labeledData: list) {
            double z = model.dot(labeledData.data);
            double score = 1.0 / (1.0 + Math.exp(-z));
            if (score >= 0.5 && labeledData.label == 1)
                N_RIGHT ++;
            if (score < 0.5 && labeledData.label == -1)
                N_RIGHT ++;
            N_TOTAL ++;
        }
        return 1.0 * N_RIGHT / N_TOTAL;
    }
    public double auc(List<LabeledData> list, DenseVector model) {
        int length = list.size();
        double[] scores = new double[length];
        double[] labels = new double[length];

        int cnt = 0;
        for (LabeledData labeledData: list) {
            double z = model.dot(labeledData.data);
            double score = 1.0 / (1.0 + Math.exp(-z));

            scores[cnt] = score;
            labels[cnt] = labeledData.label;
            cnt ++;
        }

        Sort.quickSort(scores, labels, 0, length, new DoubleComparator() {

            public int compare(double i, double i1) {
                if (Math.abs(i - i1) < 10e-12) {
                    return 0;
                } else {
                    return i - i1 > 10e-12 ? 1 : -1;
                }
            }

            public int compare(Double o1, Double o2) {
                if (Math.abs(o1 - o2) < 10e-12) {
                    return 0;
                } else {
                    return o1 - o2 > 10e-12 ? 1 : -1;
                }
            }
        });

        long M = 0, N = 0;
        for (int i = 0; i < scores.length; i ++) {
            if (labels[i] == 1.0)
                M ++;
            else
                N ++;
        }

        double sigma = 0.0;
        for (long i = M + N - 1; i >= 0; i --) {
            if (labels[(int) i] == 1.0) {
                sigma += i;
            }
        }

        double auc = (sigma - (M + 1) * M / 2) / (M * N);
        System.out.println("sigma=" + sigma + " M=" + M + " N=" + N);
        return auc;
    }
    public double lossFunctionValue(List<LabeledData> labeledData,
                                    DenseVector model, double lambda){
        double result = 0;
        for(LabeledData l: labeledData){
            double exp = Math.exp(- l.label * model.dot(l.data));
            double loss = Math.log(1 + exp);
            result += loss;
        }
        for(Double w : model.values){
            result += lambda * Math.abs(w);
        }
        return result;
    }

    public double getVectorLength(DenseVector model){
        double length = 0;
        for(Double w: model.values){
            length += w * w;
        }
        return Math.sqrt(length);
    }
    public double logLoss(List<LabeledData> list, DenseVector model_x, DenseVector model_z, double lambda) {
        double loss = 0.0;
        for (LabeledData labeledData: list) {
            double p = model_x.dot(labeledData.data);
            double z = p * labeledData.label;
            loss += Math.log(1 + Math.exp(-z));
        }
        for(Double v : model_z.values){
            loss += lambda * (v > 0? v : -v);
        }

        return loss;
    }
    public boolean converge(DenseVector oldModel, DenseVector newModel){
        double delta = 0;
        for(int i = 0; i < oldModel.values.length; i++){
            delta += Math.pow(oldModel.values[i] - newModel.values[i], 2);
        }
        System.out.println("This iteration average changes " + delta);
        if(delta < 0.01){
            return true;
        }else{
            return false;
        }
    }
}
