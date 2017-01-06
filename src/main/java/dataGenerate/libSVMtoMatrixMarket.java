package dataGenerate;

import Utils.LabeledData;

import java.io.*;
import java.util.Arrays;
import java.util.Random;

/**
 * Created by 王羚宇 on 2016/10/18.
 */
public class libSVMtoMatrixMarket {

    public static void main(String[] argv) throws IOException {
        System.out.print("Usage: Path sampleNum featureDimension");
        String path = argv[0], line;
        int M = Integer.parseInt(argv[1]);
        int N = Integer.parseInt(argv[2]);

        BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
        File matrixFile =new File(path + "matrix");
        File valueFile = new File(path + "value");

        int nonZeroCount = 0, lineNumberCount = 0;
        while ((line = reader.readLine()) != null) {
            String[] parts = line.split(" ");
            nonZeroCount += parts.length - 1;
            lineNumberCount++;
        }
        reader = new BufferedReader(new FileReader(new File(path)));
        if(!matrixFile.exists()){
            matrixFile.createNewFile();
        }
        if(!valueFile.exists()){
            valueFile.createNewFile();
        }
        //true = append file
        FileWriter matrixFileWritter = new FileWriter(matrixFile.getName(),false);
        FileWriter valueFileWriter = new FileWriter(valueFile.getName(), false);
        BufferedWriter matrixBufferWritter = new BufferedWriter(matrixFileWritter);
        BufferedWriter valueBufferWriter = new BufferedWriter(valueFileWriter);
        matrixBufferWritter.write("%%MatrixMarket matrix coordinate real symmetric\n" +
                "%\n" +
                "%  GR_30_30\n" +
                "%  Finite difference laplacian, from the\n" +
                "%  LAPLACIANS group of the Harwell-Boeing Sparse Matrix Collection.\n" +
                "%\n");
        valueBufferWriter.write("%%MatrixMarket matrix coordinate real symmetric\n" +
                "%\n" +
                "%  GR_30_30\n" +
                "%  Finite difference laplacian, from the\n" +
                "%  LAPLACIANS group of the Harwell-Boeing Sparse Matrix Collection.\n" +
                "%\n");
        matrixBufferWritter.write(M + " " + N + " " + nonZeroCount+"\n");
        valueBufferWriter.write(M + " " + 1 + " " + lineNumberCount+"\n");
        int lineCount = 1;
        while ((line = reader.readLine()) != null) {
            String[] parts = line.split(" ");
            double label = Double.parseDouble(parts[0]);
            if (label == 0)
                label = -1;
            valueBufferWriter.write(lineCount + " 1 " + label + "\n");
            int length = parts.length - 1;
            int[] indices = new int[length];
            double[] values = new double[length];
            for (int i = 0; i < length; i ++) {
                String kv = parts[i + 1];
                String[] kvParts = kv.split(":");
                int idx = Integer.parseInt(kvParts[0]);
                double value = Double.parseDouble(kvParts[1]);
                indices[i] = idx;
                values[i]  = value;
                matrixBufferWritter.write(lineCount + " " + indices[i] + " " + values[i] + "\n");
            }
            lineCount ++;
        }
        matrixBufferWritter.close();
        valueBufferWriter.close();
    }
}
