package dataGenerate;

import java.util.Arrays;
import java.util.Random;

/**
 * Created by 王羚宇 on 2016/10/18.
 */
public class linearDatasetGenerateNew {

    public static void main(String[] argv)
    {
        Random random = new Random(System.currentTimeMillis());
        //System.out.println("Usage: dataGenerate.linearDatasetGenerate featureNum sampleNum density");
        int featureDimension = Integer.parseInt(argv[0]);
        int sampleNumber = Integer.parseInt(argv[1]);
        double density = Double.parseDouble(argv[2]);
        if(density > 1 || density <= 0){
            System.out.println("Density need to be in 0 - 1.");
        }
        double realWeights[] = new double[featureDimension];
        double tuple[] = new double[featureDimension];
        double tupleValue;
        Arrays.fill(tuple, 0);
        Arrays.fill(realWeights, 0);
        for(int i = 0; i < featureDimension; i++){
            realWeights[i] = random.nextDouble() % 50;
        }
        for(int i = 0; i < sampleNumber; i++){
            tupleValue = 0;
            for(int j = 0; j < featureDimension; j++){
                if(random.nextDouble() <= density) {
                    tuple[j] = random.nextDouble() % 50;
                }
            }
            tupleValue += random.nextGaussian();
            System.out.print(tupleValue);
            for(int j = 0; j < featureDimension; j++){
                if(tuple[j] != 0){
                    System.out.print(" ");
                    System.out.print(j);
                    System.out.print(":");
                    System.out.print(tuple[j]);
                }
                tuple[j] = 0;
            }
            System.out.println();
        }
    }
}
