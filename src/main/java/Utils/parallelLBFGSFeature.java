package Utils;

import math.DenseVector;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by leleyu on 2015/9/24.
 */
public class parallelLBFGSFeature {

    private static final Log LOG = LogFactory.getLog(parallelLBFGSFeature.class);

    private static int threadNum;
    private static double lambda;

    private static double changesOfX(double[] oldX, double [] newX){
        double result = 0.0D;
        for(int i = 0; i < oldX.length; i++){
            result += (oldX[i] - newX[i]) * (oldX[i] - newX[i]);
        }
        result = Math.sqrt(result) / oldX.length;
        return result;
    }
    public static void train(ADMMFeatureState state,
                             int maxIterNum,
                             int lbfgshistory,
                             int threadNum,
                             double rhoADMM,
                             double lambda,
                             int iterationADMM,
                             List<LabeledData> trainCorpus,
                             String algorithm,
                             DenseVector z) {

        parallelLBFGSFeature.lambda = lambda;

        int localFeatureNum = state.featureDimension;
        parallelLBFGSFeature.threadNum = threadNum;

        double[] xx    = new double[localFeatureNum];
        double[] xNew  = new double[localFeatureNum];
        System.arraycopy(state.x.values, 0, xx, 0, localFeatureNum);
        System.arraycopy(xx, 0, xNew, 0, localFeatureNum);

        double[] g     = new double[localFeatureNum];
        double[] gNew  = new double[localFeatureNum];

        double[] dir = new double[localFeatureNum];

        Arrays.fill(dir, 0);
        ArrayList<double []> s = new ArrayList<double []>();
        ArrayList<double []> y = new ArrayList<double []>();
        ArrayList<Double> rhoLBFGS = new ArrayList<Double>();

        int iter = 1;

        double loss = getGradientLoss(state, xx, rhoADMM, g, z.values, trainCorpus, algorithm) ;
        System.arraycopy(g, 0, gNew, 0, localFeatureNum);

        double[] xtmp = new double[localFeatureNum];
        while (iter < maxIterNum) {
            twoLoop(s, y, rhoLBFGS, g, localFeatureNum, dir);

            loss = linearSearch(xx, xNew, dir, gNew, loss, iter, state, rhoADMM, z.values, trainCorpus, algorithm) ;

            String infoMsg = "state feature num=" + state.featureDimension + " admm iteration=" + iterationADMM
                    + " lbfgs iteration=" + iter + " loss=" + loss;
            //LOG.info(infoMsg);

            shift(localFeatureNum, lbfgshistory, xx, xNew, g, gNew, s, y, rhoLBFGS);

            iter ++;
            if(iter > 2 && changesOfX(xx, xtmp) < 1e-4){
                break;
            }
            System.arraycopy(xx, 0, xtmp, 0, localFeatureNum);
        }

        System.arraycopy(xx, 0, state.x.values, 0, localFeatureNum);
        System.out.println("Value " + getLoss(state, xx, rhoADMM, z.values, trainCorpus, "Lasso"));
    }

    private static double getGradientLoss(ADMMFeatureState state,
                                          double[] localX,
                                          double rhoADMM,
                                          double[] g,
                                          double[] z,
                                          List<LabeledData> trainCorpus,
                                          String algorithm) {
        double loss = 0.0;
        int localFeatureNum = state.featureDimension;
        Arrays.fill(g, 0);
        for (int id = 0; id < trainCorpus.size(); id++) {
            LabeledData l = trainCorpus.get(id);
            double AX = 0;
            for(int i = 0; i < l.data.indices.length; i++){
                AX += (l.data.values == null? 1:l.data.values[i]) * localX[l.data.indices[i]];
            }
            double score = AX - state.AX[id] - z[id] + state.globalAX[id] + state.u.values[id];
            loss += score * score * rhoADMM / 2.0;
            for(int i = 0; i < l.data.indices.length; i++){
                g[l.data.indices[i]] += rhoADMM * score * (l.data.values == null? 1:l.data.values[i]);
            }
        }
        for (int i = 0; i < localFeatureNum; i ++) {
            if(algorithm.equals("Lasso") || algorithm.equals("LogisticRegression")){
                if(localX[i] > 0){
                    g[i] += lambda;
                }else if(localX[i] < 0){
                    g[i] += lambda;
                }
                loss += lambda * Math.abs(localX[i]);
            }else if(algorithm.equals("SVM")){
                g[i] += 2 * lambda * localX[i];
                loss += lambda * localX[i] * localX[i];
            }
        }
        return loss;
    }
    private static double getLoss(ADMMFeatureState state,
                                  double[] localX,
                                  double rhoADMM,
                                  double[] z,
                                  List<LabeledData> trainCorpus,
                                  String algorithm) {
        double loss = 0.0;
        int localFeatureNum = state.featureDimension;
        for (int id = 0; id < trainCorpus.size(); id++) {
            LabeledData l = trainCorpus.get(id);
            double AX = 0;
            for(int i = 0; i < l.data.indices.length; i++){
                AX += (l.data.values == null? 1:l.data.values[i]) * localX[l.data.indices[i]];
            }
            double score = AX - state.AX[id] - z[id] + state.globalAX[id] + state.u.values[id];
            loss += score * score * rhoADMM / 2.0;
        }
        for (int i = 0; i < localFeatureNum; i ++) {
            if(algorithm.equals("Lasso")  || algorithm.equals("LogisticRegression") ){
                loss += lambda * Math.abs(localX[i]);
            }else if(algorithm.equals("SVM")){
                loss += lambda * localX[i] * localX[i];
            }
        }
        return loss;
    }

    private static void twoLoop(ArrayList<double[]> s,
                                ArrayList<double[]> y,
                                ArrayList<Double> rhoLBFGS,
                                double[] g,
                                int localFeatureNum,
                                double[] dir) {

        times(dir, g, -1, localFeatureNum);

        int count = s.size();
        if (count != 0) {
            double[] alphas = new double[count];
            for (int i = count - 1; i >= 0; i --) {
                alphas[i] = -dot(s.get(i), dir, localFeatureNum) / rhoLBFGS.get(i);
                timesBy(dir, y.get(i), alphas[i], localFeatureNum);
            }

            double yDotY = dot(y.get(y.size() - 1), y.get(y.size() - 1), localFeatureNum);
            double scalar = rhoLBFGS.get(rhoLBFGS.size() - 1) / yDotY;
            timesBy(dir, scalar, localFeatureNum);

            for (int i = 0; i < count; i ++) {
                double beta = dot(y.get(i), dir, localFeatureNum) / rhoLBFGS.get(i);
                timesBy(dir, s.get(i), -alphas[i] - beta, localFeatureNum);
            }
        }

    }

    private static double linearSearch(double[] x,
                                       double[] xNew,
                                       double[] dir,
                                       double[] gNew,
                                       double oldLoss,
                                       int iteration,
                                       ADMMFeatureState state,
                                       double rhoADMM,
                                       double[] z,
                                       List<LabeledData> trainCorpus,
                                       String algorithm) {

        int localFeatureNum = state.featureDimension;

        double loss = Double.MAX_VALUE;
        double origDirDeriv = dot(dir, gNew, localFeatureNum);

        // if a non-descent direction is chosen, the line search will break anyway, so throw here
        // The most likely reason for this is a bug in your function's gradient computation
        if (origDirDeriv >= 0) {
            LOG.info("L-BFGS chose a non-descent direction, check your gradient!");
            return 0.0;
        }
        if(Double.isNaN(origDirDeriv)){
            LOG.info("NaN happens!");
        }

        double alpha = 1.0;
        double backoff = 0.5;
        if (iteration == 1) {
            alpha = 1 / Math.sqrt(dot(dir, dir, localFeatureNum));
            backoff = 0.1;
        }

        double c1 = 1e-4;
        int i = 1, step = 20;

        while ((loss > oldLoss + c1 * origDirDeriv * alpha) && (step > 0)) {
            timesBy(xNew, x, dir, alpha, localFeatureNum);
            loss = getLoss(state, xNew, rhoADMM, z, trainCorpus, algorithm) ;
            String infoMsg = "state feature num=" + state.featureDimension + " lbfgs iteration=" + iteration
                    + " line search iteration=" + i + " end loss=" + loss + " alpha=" + alpha
                    + " oldloss=" + oldLoss + " delta=" + (c1*origDirDeriv*alpha) + " origDirDeriv=" + origDirDeriv;
            //LOG.info(infoMsg);
            alpha *= backoff;
            i ++;
            step -= 1;
        }
        getGradientLoss(state, xNew, rhoADMM, gNew, z, trainCorpus, algorithm);
        return loss;
    }

    private static void shift(int localFeatureNum,
                              int lbfgsHistory,
                              double[] x,
                              double[] xNew,
                              double[] g,
                              double[] gNew,
                              ArrayList<double []> s,
                              ArrayList<double []> y,
                              ArrayList<Double> rhoLBFGS) {
        int length = s.size();

        if (length < lbfgsHistory) {
            s.add(new double[localFeatureNum]);
            y.add(new double[localFeatureNum]);
        } else {
            double [] temp = s.remove(0);
            s.add(temp);
            temp = y.remove(0);
            y.add(temp);
            rhoLBFGS.remove(0);
        }

        double [] lastS = s.get(s.size() - 1);
        double [] lastY = y.get(y.size() - 1);

        timesBy(lastS, xNew, x, -1, localFeatureNum);
        timesBy(lastY, gNew, g, -1, localFeatureNum);
        double rho = dot(lastS, lastY, localFeatureNum);
        rhoLBFGS.add(rho);

        System.arraycopy(xNew, 0, x, 0, localFeatureNum);
        System.arraycopy(gNew, 0, g, 0, localFeatureNum);
    }

    private static void times(double [] a, double [] b, double x, int length) {
        for (int i = 0; i < length; i ++) {
            if(Double.isNaN(a[i]) || Double.isInfinite(a[i])){
                LOG.info("NaN a[i] happens!");
            }
            if(Double.isNaN(b[i]) || Double.isInfinite(b[i])){
                LOG.info("NaN b[i] happens!");
            }
            if(Double.isNaN(x) || Double.isInfinite(x)){
                LOG.info("NaN b[i] happens!");
            }
            a[i] = b[i] * x;
        }
    }

    private static void timesBy(double [] a, double [] b, double x, int length) {
        for (int i = 0; i < length; i ++){
            if(Double.isNaN(a[i]) || Double.isInfinite(a[i])){
                LOG.info("NaN a[i] happens!");
            }
            if(Double.isNaN(b[i]) || Double.isInfinite(b[i])){
                LOG.info("NaN b[i] happens!");
            }
            if(Double.isNaN(x) || Double.isInfinite(x)){
                LOG.info("NaN b[i] happens!");
            }
            a[i] += b[i] * x;
        }

    }

    private static void timesBy(double [] a, double [] b, double [] c, double x, int length) {
        for (int i = 0; i < length; i ++) {
            if(Double.isNaN(a[i]) || Double.isInfinite(a[i])){
                LOG.info("NaN a[i] happens!");
            }
            if(Double.isNaN(b[i]) || Double.isInfinite(b[i])){
                LOG.info("NaN b[i] happens!");
            }
            if(Double.isNaN(c[i]) || Double.isInfinite(c[i])){
                LOG.info("NaN b[i] happens!");
            }
            a[i] = b[i] + c[i] * x;
        }
    }

    private static void timesBy(double [] a, double x, int length) {
        for (int i = 0; i < length; i ++)
            a[i] *= x;
    }

    private static double dot(double [] a, double [] b, int length) {
        double ret = 0.0;
        for (int i = 0; i < length; i ++){
            if(Double.isNaN(a[i]) || Double.isInfinite(a[i])){
                LOG.info("NaN a[i] happens!");
            }
            if(Double.isNaN(b[i]) || Double.isInfinite(b[i])){
                LOG.info("NaN b[i] happens!");
            }
            ret += a[i] * b[i];
        }
        return ret;
    }
}
