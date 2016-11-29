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
        assert filesList != null;
        for(int i = 0; i < filesList.length; i++){
            if(filesList[i].isFile() && filesList[i].getName().endsWith(".log")){
                if(filesList[i].getName().startsWith("SVM") || filesList[i].getName().startsWith("LR")) {
                    File loss = new File(filesList[i].getName() + "loss");
                    File trainAuc = new File(filesList[i].getName() + "trainAuc");
                    File testAuc = new File(filesList[i].getName() + "testAuc");
                    File changes = new File(filesList[i].getName() + "changes");
                    FileWriter lossWriter = new FileWriter(loss);
                    FileWriter trainAucWriter = new FileWriter(trainAuc);
                    FileWriter testAucWriter = new FileWriter(testAuc);
                    FileWriter changesWriter = new FileWriter(changes);

                    BufferedWriter lossBufferWriter = new BufferedWriter(lossWriter);
                    BufferedWriter trainAucBufferWriter = new BufferedWriter(trainAucWriter);
                    BufferedWriter testAucBufferWriter = new BufferedWriter(testAucWriter);
                    BufferedWriter changesBufferWriter = new BufferedWriter(changesWriter);

                    BufferedReader reader = new BufferedReader(new FileReader(filesList[i]));
                    String tmpString;
                    while ((tmpString = reader.readLine()) != null) {
                        if (tmpString.startsWith("loss=")) {
                            String[] splits = tmpString.replace("=", " ").split(" ");
                            lossBufferWriter.write(splits[1] + "\n");
                            trainAucBufferWriter.write(splits[3] + "\n");
                            testAucBufferWriter.write(splits[5] + "\n");
                        }
                        if (tmpString.startsWith("This iteration average")) {
                            String[] splits = tmpString.replace("=", " ").split(" ");
                            changesBufferWriter.write(splits[4] + "\n");
                        }
                    }
                    lossBufferWriter.close();lossWriter.close();
                    trainAucBufferWriter.close();trainAucWriter.close();
                    testAucBufferWriter.close();testAucWriter.close();
                    changesBufferWriter.close();changesWriter.close();
                }else if(filesList[i].getName().startsWith("Lasso") || filesList[i].getName().startsWith("Linear")) {
                    File loss = new File(filesList[i].getName() + "loss");
                    File testResidual = new File(filesList[i].getName() + "residual");
                    File changes = new File(filesList[i].getName() + "changes");

                    FileWriter lossWriter = new FileWriter(loss);
                    FileWriter testResidualWriter = new FileWriter(testResidual);
                    FileWriter changesWriter = new FileWriter(changes);
                    BufferedWriter lossBufferWriter = new BufferedWriter(lossWriter);
                    BufferedWriter testResidualBufferWriter = new BufferedWriter(testResidualWriter);
                    BufferedWriter changesBufferWriter = new BufferedWriter(changesWriter);

                    BufferedReader reader = new BufferedReader(new FileReader(filesList[i]));
                    String tmpString;
                    while ((tmpString = reader.readLine()) != null) {
                        if (tmpString.startsWith("loss=")) {
                            String[] splits = tmpString.replace("=", " ").split(" ");
                            lossBufferWriter.write(splits[1] + "\n");
                            testResidualBufferWriter.write(splits[3] + "\n");
                        }
                        if (tmpString.startsWith("This iteration average")) {
                            String[] splits = tmpString.replace("=", " ").split(" ");
                            changesBufferWriter.write(splits[4] + "\n");
                        }
                    }
                    lossBufferWriter.close();lossWriter.close();
                    testResidualBufferWriter.close();testResidualWriter.close();
                    changesBufferWriter.close();changesWriter.close();
                }
            }
        }
    }
}
