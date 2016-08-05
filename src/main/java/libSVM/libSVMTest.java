package libSVM;

import java.io.File;
import java.io.IOException;

import de.bwaldvogel.liblinear.*;

/**
 * Created by 王羚宇 on 2016/8/5.
 */
public class libSVMTest {
    public static void main(String[] argv) throws InvalidInputDataException, IOException{
        System.out.println("libSVM.libSVMTest datafile lambda");
        File dataFile = new File(argv[0]);
        double lambda = Double.parseDouble(argv[1]);
        double C = 1.0 / (2.0 * lambda);    // cost of constraints violation
        double eps = 0.01; // stopping criteria
        Problem problem = new Problem();
        SolverType solver = SolverType.L2R_L1LOSS_SVC_DUAL;
        Parameter parameter = new Parameter(solver, C, eps);
        Problem.readFromFile(dataFile, -1);
        double []target = new double[problem.l];
        Linear.crossValidation(problem, parameter, 2, target);
        //Model model = Linear.train(problem, parameter);
        //model.save(new File("demo.model"));
    }
}
