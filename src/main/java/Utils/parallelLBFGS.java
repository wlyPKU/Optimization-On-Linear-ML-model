package Utils;
import math.DenseVector;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Created by leleyu on 2015/9/24.
 */
public class parallelLBFGS {

    private static final Log LOG = LogFactory.getLog(parallelLBFGS.class);

    private static int threadNum;
    public static void train(ADMMState state,
                             int maxIterNum,
                             int lbfgshistory,
                             int threadNum,
                             double rhoADMM,
                             int iterationADMM,
                             List<LabeledData> trainCorpus,
                             String algorithm,
                             DenseVector z) {

        int localFeatureNum = state.featureNum;
        parallelLBFGS.threadNum = threadNum;

        double[] xx    = new double[localFeatureNum];
        double[] xNew  = new double[localFeatureNum];
        System.arraycopy(state.x.values, 0, xx, 0, localFeatureNum);
        System.arraycopy(xx, 0, xNew, 0, localFeatureNum);

        double[] g     = new double[localFeatureNum];
        double[] gNew  = new double[localFeatureNum];

        double[] dir = new double[localFeatureNum];

        ArrayList<double []> s = new ArrayList<double []>();
        ArrayList<double []> y = new ArrayList<double []>();
        ArrayList<Double> rhoLBFGS = new ArrayList<Double>();

        int iter = 1;

        double loss = getGradientLoss(state, xx, rhoADMM, g, z.values, trainCorpus, algorithm) ;
        System.arraycopy(g, 0, gNew, 0, localFeatureNum);
        while (iter < maxIterNum) {
            twoLoop(s, y, rhoLBFGS, g, localFeatureNum, dir);

            loss = linearSearch(xx, xNew, dir, gNew, loss, iter, state, rhoADMM, z.values, trainCorpus, algorithm) ;

            String infoMsg = "state feature num=" + state.featureNum + " admm iteration=" + iterationADMM
                    + " lbfgs iteration=" + iter + " loss=" + loss;
            //LOG.info(infoMsg);

            shift(localFeatureNum, lbfgshistory, xx, xNew, g, gNew, s, y, rhoLBFGS);

            iter ++;
        }

        System.arraycopy(xx, 0, state.x.values, 0, localFeatureNum);
    }

    private static double getGradientLoss(ADMMState state,
                                          double[] localX,
                                          double rhoADMM,
                                          double[] g,
                                          double[] z,
                                          List<LabeledData> trainCorpus,
                                          String algorithm) {
        double loss = 0.0;

        int localFeatureNum = state.featureNum;

        for (int i = 0; i < localFeatureNum; i ++) {
            double temp = localX[i] - z[i] + state.u.values[i];
            g[i] = rhoADMM * temp ;
            loss += 0.5 * rhoADMM * temp * temp;
        }
        for (LabeledData l : trainCorpus) {
            if(algorithm.equals("LogisticRegression")) {
                double score = 0;
                int index[] = l.data.indices;
                for (int i = 0; i < index.length; i++) {
                    if (l.data.values != null) {
                        score += localX[index[i]] * l.data.values[i];
                    } else {
                        score += localX[index[i]];
                    }
                }
                score *= l.label;
                loss += Math.log(1.0 + Math.exp(-score));
                double gradient = - Math.exp(-score) / (1.0 + Math.exp(-score)) * l.label;
                for (int i = 0; i < l.data.indices.length; i++) {
                    if (l.data.values == null) {
                        g[l.data.indices[i]] += gradient;
                    } else {
                        g[l.data.indices[i]] += gradient * l.data.values[i];
                    }
                }
            }else if(algorithm.equals("SVM")){
                double score = 0;
                int index[] = l.data.indices;
                for (int i = 0; i < index.length; i++) {
                    if (l.data.values != null) {
                        score += localX[index[i]] * l.data.values[i];
                    } else {
                        score += localX[index[i]];
                    }
                }
                score = 1 - l.label * score;

                if(score > 0)
                {
                    loss += score;
                    for (int i = 0; i < index.length; i++) {
                        if (l.data.values == null) {
                            g[index[i]] -= l.label;
                        } else {
                            g[index[i]] -= l.data.values[i] * l.label;
                        }
                    }
                }
            }else if(algorithm.equals("Lasso") || algorithm.equals("LinearRegression")){
                double score = 0;
                for (int i = 0; i < l.data.indices.length; i++) {
                    if (l.data.values != null) {
                        score += localX[l.data.indices[i]] * l.data.values[i];
                    } else {
                        score += localX[l.data.indices[i]];
                    }
                }
                score = score - l.label;
                loss += 0.5 * score * score;
                for (int i = 0; i < l.data.indices.length; i++) {
                    if (l.data.values == null) {
                        g[l.data.indices[i]] += score;
                    } else {
                        g[l.data.indices[i]] += l.data.values[i] * score;
                    }
                }
            }
        }
        return loss;
    }
    private static double getLoss(ADMMState state,
                                  double[] localX,
                                  double rhoADMM,
                                  double[] z,
                                  List<LabeledData> trainCorpus,
                                  String algorithm) {
        double loss = 0.0;

        int localFeatureNum = state.featureNum;

        for (int i = 0; i < localFeatureNum; i ++) {
            double temp = localX[i] - z[i] + state.u.values[i];
            loss += 0.5 * rhoADMM * temp * temp;
        }

        Iterator<LabeledData> iter = trainCorpus.iterator();
        while (iter.hasNext()) {
            if(algorithm.equals("LogisticRegression")) {
                LabeledData l = iter.next();
                double score = 0;
                int[] indices = l.data.indices;
                for (int i = 0; i < indices.length; i++) {
                    if (l.data.values != null) {
                        score += localX[indices[i]] * l.data.values[i];
                    } else {
                        score += localX[indices[i]];
                    }
                }
                score *= l.label;
                loss += Math.log(1.0 + Math.exp(-score));
            }else if(algorithm.equals("SVM")){
                LabeledData l = iter.next();
                double score = 0;
                for (int i = 0; i < l.data.indices.length; i++) {
                    if (l.data.values != null) {
                        score += localX[l.data.indices[i]] * l.data.values[i];
                    } else {
                        score += localX[l.data.indices[i]];
                    }
                }
                score = 1 - l.label * score;
                if(score > 0){
                    loss += score;
                }
            }else if(algorithm.equals("Lasso") || algorithm.equals("LinearRegression")){
                LabeledData l = iter.next();
                double score = 0;
                for (int i = 0; i < l.data.indices.length; i++) {
                    if (l.data.values != null) {
                        score += localX[l.data.indices[i]] * l.data.values[i];
                    } else {
                        score += localX[l.data.indices[i]];
                    }
                }
                score = l.label - score;
                loss += 0.5 * score * score;
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
                                       ADMMState state,
                                       double rhoADMM,
                                       double[] z,
                                       List<LabeledData> trainCorpus,
                                       String algorithm) {

        int localFeatureNum = state.featureNum;

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
            return 0.0;
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
            String infoMsg = "state feature num=" + state.featureNum + " lbfgs iteration=" + iteration
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
        for (int i = 0; i < length; i ++)
            a[i] = b[i] * x;
    }

    private static void timesBy(double [] a, double [] b, double x, int length) {
        for (int i = 0; i < length; i ++)
            a[i] += b[i] * x;
    }

    private static void timesBy(double [] a, double [] b, double [] c, double x, int length) {
        for (int i = 0; i < length; i ++)
            a[i] = b[i] + c[i] * x;
    }

    private static void timesBy(double [] a, double x, int length) {
        for (int i = 0; i < length; i ++)
            a[i] *= x;
    }

    private static double dot(double [] a, double [] b, int length) {
        double ret = 0.0;
        for (int i = 0; i < length; i ++){
            ret += a[i] * b[i];
        }
        return ret;
    }
}
