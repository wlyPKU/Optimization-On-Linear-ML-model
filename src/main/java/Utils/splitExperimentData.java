package Utils;

import java.awt.*;
import java.io.*;

/**
 * Created by 王羚宇 on 2016/8/8.
 */
public class splitExperimentData {
    public static void main(String args[]) throws IOException{
        String inputDir = args[0];
        File file = new File(inputDir);
        File[] filesList = file.listFiles();
        for(int i = 0; i < filesList.length; i++){
            if(filesList[i].isFile() && filesList[i].getName().endsWith(".log")){
                if(filesList[i].getName().startsWith("SVM") || filesList[i].getName().startsWith("LR")) {
                    File loss = new File(filesList[i].getName() + "loss");
                    File trainAuc = new File(filesList[i].getName() + "trainAuc");
                    File testAuc = new File(filesList[i].getName() + "testAuc");
                    File trainAccuracy = new File(filesList[i].getName() + "trainAccuracy");
                    File testAccuracy = new File(filesList[i].getName() + "testAccuracy");
                    FileWriter lossWriter = new FileWriter(loss);
                    FileWriter trainAucWriter = new FileWriter(trainAuc);
                    FileWriter testAucWriter = new FileWriter(testAuc);
                    FileWriter trainAccuracyWriter = new FileWriter(trainAccuracy);
                    FileWriter testAccuracyWriter = new FileWriter(testAccuracy);
                    BufferedWriter lossBufferWriter = new BufferedWriter(lossWriter);
                    BufferedWriter trainAucBufferWriter = new BufferedWriter(trainAucWriter);
                    BufferedWriter testAucBufferWriter = new BufferedWriter(testAucWriter);
                    BufferedWriter trainAccuracyBufferWriter = new BufferedWriter(trainAccuracyWriter);
                    BufferedWriter testAccuracyBufferWriter = new BufferedWriter(testAccuracyWriter);

                    BufferedReader reader = new BufferedReader(new FileReader(filesList[i]));
                    String tmpString;
                    while ((tmpString = reader.readLine()) != null) {
                        if (tmpString.startsWith("loss=")) {
                            String[] splits = tmpString.replace("=", " ").split(" ");
                            lossBufferWriter.write(splits[1] + "\n");
                            trainAucBufferWriter.write(splits[3] + "\n");
                            testAucBufferWriter.write(splits[5] + "\n");
                            trainAccuracyBufferWriter.write(splits[7] + "\n");
                            testAccuracyBufferWriter.write(splits[9] + "\n");
                        }
                    }
                    lossBufferWriter.close();lossWriter.close();
                    trainAccuracyBufferWriter.close();trainAccuracyWriter.close();
                    testAccuracyBufferWriter.close();testAccuracyWriter.close();
                    trainAucBufferWriter.close();trainAucWriter.close();
                    testAucBufferWriter.close();testAucWriter.close();
                }

            }
        }
    }
}
