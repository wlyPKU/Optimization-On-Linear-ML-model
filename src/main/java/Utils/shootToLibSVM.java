package Utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

/**
 * Created by 王羚宇 on 2016/7/29.
 */
public class shootToLibSVM {
    public static void main(String[] argv) throws Exception {
        String path = argv[0];
        int sampleNum = Integer.parseInt(argv[1]);
        int nonZero = Integer.parseInt(argv[2]);
        int startLine = 0;
        while(sampleNum - startLine > 0){
            int handlePerLoop = 10000;
            String []line = new String[handlePerLoop];
            for(int i = 0; i < handlePerLoop; i++){
                line[i] = "";
            }
            handlePerLoop = Math.min(sampleNum - startLine, handlePerLoop);
            String []label = new String[handlePerLoop];
            BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
            String readData;
            int cnt = 0;
            int labelCount = 0;
            while ((readData = reader.readLine()) != null) {
                if(readData.startsWith("#")){
                    continue;
                }
                if(cnt < nonZero){
                    String []lineData = readData.split(" ");
                    int sampleIndex = Integer.parseInt(lineData[0]);
                    int featureIndex = Integer.parseInt(lineData[1]);
                    double featureValue = Double.parseDouble(lineData[2]);
                    if(sampleIndex >= startLine && sampleIndex < startLine + handlePerLoop){
                        line[sampleIndex - startLine] += " " + featureIndex + ":" + featureValue;
                    }
                }else{
                    if(labelCount >= startLine && labelCount <  startLine + handlePerLoop) {
                        String lineData = readData.replace(" ", "");
                        label[labelCount - startLine] = lineData;
                    }
                    labelCount ++;
                }
                cnt++;
            }
            for(int i = 0; i < sampleNum - startLine && i < handlePerLoop; i++){
                if(label[i] == null){
                    label[i] = "0";
                }
                System.out.println(label[i] + line[i]);
            }
            startLine += handlePerLoop;
        }
    }
}
