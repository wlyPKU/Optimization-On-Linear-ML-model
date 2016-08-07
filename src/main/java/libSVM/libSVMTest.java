package libSVM;

import java.io.*;
import java.util.HashSet;

import de.bwaldvogel.liblinear.*;

/**
 * Created by 王羚宇 on 2016/8/5.
 */
public class libSVMTest {
    public static HashSet randomPick(int totalSize, double trainRatio){
        int size = (int)(totalSize * trainRatio);
        HashSet<Integer> result = new HashSet<Integer>();
        for(int i = 0; i < size; i++){
            int random = (int)(Math.random() * totalSize);
            result.add(random);
        }
        while(result.size() < size){
            int random = (int)(Math.random() * totalSize);
            result.add(random);
        }
        return result;
    }

    public static void splitData(double trainRatio, String fileName) throws IOException{
        int sampleCount = 0;
        File f = new File(fileName);
        InputStream input = new FileInputStream(f);
        BufferedReader b = new BufferedReader(new InputStreamReader(input));
        String value = b.readLine();
        while(value != null) {
            sampleCount++;
            value = b.readLine();
        }
        b.close();
        input.close();

        File trainFile = new File(fileName + "train.data");
        File testFile = new File(fileName + "test.data");

        FileWriter trainWriter = new FileWriter(trainFile);
        FileWriter testWriter = new FileWriter(testFile);
        BufferedWriter trainBufferWriter = new BufferedWriter(trainWriter);
        BufferedWriter testBufferWriter = new BufferedWriter(testWriter);

        HashSet trainSequence = randomPick(sampleCount, trainRatio);
        int lineIdx = 0;
        input = new FileInputStream(f);
        b = new BufferedReader(new InputStreamReader(input));
        value = b.readLine();
        while(value != null) {
            if(trainSequence.contains(lineIdx)){
                trainBufferWriter.write(value + "\n");
            }else{
                testBufferWriter.write(value+ "\n");
            }
            value = b.readLine();
            lineIdx++;
        }
        b.close();
        input.close();
        trainBufferWriter.close();
        trainWriter.close();
        testBufferWriter.close();
        testWriter.close();
    }

    public static double predict(Model model, Problem problem){
        double result = 0;
        int size = problem.l;
        for(int i = 0; i < size; i++){
            Feature[] f = problem.x[i];
            double realLabel = problem.y[i];
            double predictLabel = Linear.predict(model, f);
            if(realLabel * predictLabel >= 0){
                result++;
            }
        }
        return result / size;
    }
    public static void main(String[] argv) throws InvalidInputDataException, IOException{
        System.out.println("libSVM.libSVMTest datafile lambda [trainRatio]");
        String fileName = argv[0];
        double lambda = Double.parseDouble(argv[1]);
        double trainRatio = 0.5D;
        if(argv.length >= 3){
            trainRatio = Double.parseDouble(argv[2]);
            if(trainRatio >= 1|| trainRatio <= 0){
                System.out.println("[ERROR] A trainRatio belongs to (0,1) needed!");
                System.exit(1);
            }
        }
        long startPrepare = System.currentTimeMillis();
        splitData(trainRatio, fileName);
        System.out.println("[INFO]Preparing data finished ... " + (System.currentTimeMillis() - startPrepare) + " ms");
        long startRead = System.currentTimeMillis();
        double C = 1.0 / (2.0 * lambda);    // cost of constraints violation
        double eps = 0.01; // stopping criteria
        Problem trainProblem = Problem.readFromFile(new File(fileName + "train.data"), -1);
        Problem testProblem = Problem.readFromFile(new File(fileName + "test.data"), -1);
        SolverType solver = SolverType.L2R_L1LOSS_SVC_DUAL;
        Parameter parameter = new Parameter(solver, C, eps);
        System.out.println("[INFO]Reading data finished ... " + (System.currentTimeMillis() - startRead) + " ms");
        long startTrain = System.currentTimeMillis();
        Model model = Linear.train(trainProblem, parameter);
        System.out.println("[INFO]Training data finished ... " + (System.currentTimeMillis() - startTrain) + " ms");
        model.save(new File("demo.model"));
        long startTest = System.currentTimeMillis();
        double testLoss = predict(model, testProblem);
        System.out.println("[INFO]Predict data finished ... " + (System.currentTimeMillis() - startTest) + " ms");
        System.out.println("Test loss: " + testLoss);
    }
}
