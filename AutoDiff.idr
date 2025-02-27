module AutoDiff

%default total

-- FORWARD MODE IMPLEMENTATION USING DUAL NUMBERS
-- This is true automatic differentiation without symbolic rules

-- Dual numbers for forward-mode automatic differentiation
-- A dual number is a pair (a, b) representing a + bε where ε² = 0
public export
data Dual : Type where
  MkDual : Double -> Double -> Dual

-- Get the primal value of a dual number
public export
primal : Dual -> Double
primal (MkDual a _) = a

-- Get the tangent value (derivative) of a dual number
public export
tangent : Dual -> Double
tangent (MkDual _ b) = b

-- Create a dual number representing a variable (derivative = 1)
public export
mkVariable : Double -> Dual
mkVariable x = MkDual x 1.0

-- Create a dual number representing a constant (derivative = 0)
public export
mkConstant : Double -> Dual
mkConstant x = MkDual x 0.0

-- Numeric operations on dual numbers
public export
implementation Num Dual where
  -- Addition: (a, b) + (c, d) = (a+c, b+d)
  (MkDual a b) + (MkDual c d) = MkDual (a + c) (b + d)
  
  -- Multiplication: (a, b) * (c, d) = (a*c, a*d + b*c)
  (MkDual a b) * (MkDual c d) = MkDual (a * c) (a * d + b * c)
  
  -- FromInteger: n = (n, 0)
  fromInteger n = MkDual (fromInteger n) 0.0

public export
implementation Neg Dual where
  -- Negation: -(a, b) = (-a, -b)
  negate (MkDual a b) = MkDual (negate a) (negate b)
  
  -- Subtraction: (a, b) - (c, d) = (a-c, b-d)
  (MkDual a b) - (MkDual c d) = MkDual (a - c) (b - d)

public export
implementation Fractional Dual where
  -- Division: (a, b) / (c, d) = (a/c, (b*c - a*d)/(c*c))
  (MkDual a b) / (MkDual c d) = 
    MkDual (a / c) ((b * c - a * d) / (c * c))
  
  -- Reciprocal: 1/(a, b) = (1/a, -b/(a*a))
  recip (MkDual a b) = MkDual (1.0 / a) (-b / (a * a))

-- Elementary functions on dual numbers

-- sin(a + bε) = sin(a) + b*cos(a)ε
public export
sin : Dual -> Dual
sin (MkDual a b) = MkDual (prim__doubleSin a) (b * prim__doubleCos a)

-- cos(a + bε) = cos(a) - b*sin(a)ε
public export
cos : Dual -> Dual
cos (MkDual a b) = MkDual (prim__doubleCos a) (-b * prim__doubleSin a)

-- exp(a + bε) = exp(a) + b*exp(a)ε
public export
exp : Dual -> Dual
exp (MkDual a b) = 
  let expA = prim__doubleExp a
  in MkDual expA (b * expA)

-- log(a + bε) = log(a) + b/aε
public export
log : Dual -> Dual
log (MkDual a b) = 
  let safeA = if a <= 0.0 then 1.0e-10 else a
  in MkDual (prim__doubleLog safeA) (b / safeA)

-- (a + bε)^n = a^n + n*a^(n-1)*bε
public export
pow : Dual -> Double -> Dual
pow (MkDual a b) n = 
  let safeA = if a <= 0.0 && n /= cast (floor n) then 1.0e-10 else a
      powA = prim__doubleExp (n * prim__doubleLog safeA)
  in MkDual powA (n * b * prim__doubleExp ((n - 1) * prim__doubleLog safeA))

-- Square function (common operation)
public export
square : Dual -> Dual
square x = x * x

-- Compute derivative of a function at a point using forward mode
public export
forward_derivative : (Dual -> Dual) -> Double -> Double
forward_derivative f x = tangent (f (mkVariable x))

-- BACKWARD MODE IMPLEMENTATION
-- This uses computational graph-based reverse mode

-- Enhanced Expr type to support higher-order gradients
public export
data Expr = 
    ConstExpr Double
  | VarExpr Int Double      -- ID and Value
  | AddExpr Expr Expr
  | MulExpr Expr Expr
  | DivExpr Expr Expr
  | SubExpr Expr Expr
  | SinExpr Expr
  | CosExpr Expr
  | ExpExpr Expr
  | LogExpr Expr
  | PowExpr Expr Double

-- Get the value of an expression
public export
eval : Expr -> Double
eval (ConstExpr c) = c
eval (VarExpr _ v) = v
eval (AddExpr x y) = eval x + eval y
eval (MulExpr x y) = eval x * eval y
eval (DivExpr x y) = eval x / eval y
eval (SubExpr x y) = eval x - eval y
eval (SinExpr x) = prim__doubleSin (eval x)
eval (CosExpr x) = prim__doubleCos (eval x)
eval (ExpExpr x) = prim__doubleExp (eval x)
eval (LogExpr x) = prim__doubleLog (eval x)
eval (PowExpr x n) = prim__doubleExp (n * prim__doubleLog (eval x))

-- Helper function for power operation
public export
customPow : Double -> Double -> Double
customPow x n = prim__doubleExp (n * prim__doubleLog (if x <= 0.0 then 1.0e-10 else x))

-- Calculate gradient of expression with respect to variable ID
public export
grad : Expr -> Int -> Double
grad (ConstExpr _) _ = 0.0
grad (VarExpr id _) varId = if id == varId then 1.0 else 0.0
grad (AddExpr x y) varId = grad x varId + grad y varId
grad (MulExpr x y) varId = 
  grad x varId * eval y + eval x * grad y varId
grad (DivExpr x y) varId = 
  (grad x varId * eval y - eval x * grad y varId) / (eval y * eval y)
grad (SubExpr x y) varId = grad x varId - grad y varId
grad (SinExpr x) varId = prim__doubleCos (eval x) * grad x varId
grad (CosExpr x) varId = -(prim__doubleSin (eval x)) * grad x varId
grad (ExpExpr x) varId = prim__doubleExp (eval x) * grad x varId
grad (LogExpr x) varId = grad x varId / eval x
grad (PowExpr x n) varId = 
  n * customPow (eval x) (n-1.0) * grad x varId

-- Create an expression from a gradient (for higher-order differentiation)
public export
gradToExpr : Expr -> Int -> Expr
gradToExpr (ConstExpr _) _ = ConstExpr 0.0
gradToExpr (VarExpr id v) varId = 
  if id == varId then ConstExpr 1.0 else ConstExpr 0.0
gradToExpr (AddExpr x y) varId = 
  AddExpr (gradToExpr x varId) (gradToExpr y varId)
gradToExpr (MulExpr x y) varId = 
  AddExpr 
    (MulExpr (gradToExpr x varId) y)
    (MulExpr x (gradToExpr y varId))
gradToExpr (DivExpr x y) varId = 
  DivExpr 
    (SubExpr 
      (MulExpr (gradToExpr x varId) y)
      (MulExpr x (gradToExpr y varId)))
    (MulExpr y y)
gradToExpr (SubExpr x y) varId = 
  SubExpr (gradToExpr x varId) (gradToExpr y varId)
gradToExpr (SinExpr x) varId = 
  MulExpr (CosExpr x) (gradToExpr x varId)
gradToExpr (CosExpr x) varId = 
  MulExpr (ConstExpr (-1.0)) (MulExpr (SinExpr x) (gradToExpr x varId))
gradToExpr (ExpExpr x) varId = 
  MulExpr (ExpExpr x) (gradToExpr x varId)
gradToExpr (LogExpr x) varId = 
  DivExpr (gradToExpr x varId) x
gradToExpr (PowExpr x n) varId = 
  MulExpr 
    (MulExpr 
      (ConstExpr n) 
      (PowExpr x (n-1.0)))
    (gradToExpr x varId)

-- Higher-order gradients using automatic differentiation
public export
higherGrad : Nat -> Expr -> Int -> Double
higherGrad Z expr _ = eval expr
higherGrad (S Z) expr varId = grad expr varId
higherGrad (S (S n)) expr varId = 
  -- For higher-order derivatives, we differentiate the derivative expression
  let derivExpr = gradToExpr expr varId
  in higherGrad (S n) derivExpr varId

-- Create a variable for backward mode
public export
makeVar : Double -> Expr
makeVar x = VarExpr 1 x  -- Using ID 1 for the variable

-- Compute nth derivative using backward mode
public export
backward_derivative : (Expr -> Expr) -> Double -> Nat -> Double
backward_derivative f x order = higherGrad order (f (makeVar x)) 1

-- EXAMPLE FUNCTIONS

-- Example 1: f(x) = x^2 + 3x + 2
-- Forward mode
public export
example1_fwd : Dual -> Dual
example1_fwd x = square x + fromInteger 3 * x + fromInteger 2

-- Backward mode
public export
example1_back : Expr -> Expr
example1_back x = AddExpr (AddExpr (MulExpr x x) (MulExpr (ConstExpr 3.0) x)) (ConstExpr 2.0)

-- Example 2: f(x) = sin(x) * cos(x)
-- Forward mode
public export
example2_fwd : Dual -> Dual
example2_fwd x = sin x * cos x

-- Backward mode
public export
example2_back : Expr -> Expr
example2_back x = MulExpr (SinExpr x) (CosExpr x)

-- Example 3: f(x) = e^(x^2) / (1 + x^2)
-- Forward mode
public export
example3_fwd : Dual -> Dual
example3_fwd x = 
  let x_squared = square x
  in exp x_squared / (fromInteger 1 + x_squared)

-- Backward mode
public export
example3_back : Expr -> Expr
example3_back x = 
  let x_squared = MulExpr x x
  in DivExpr (ExpExpr x_squared) (AddExpr (ConstExpr 1.0) x_squared)

-- Example 4: f(x) = x^3 - 5x^2 + 7x - 3
-- Forward mode
public export
example4_fwd : Dual -> Dual
example4_fwd x = 
  let x2 = square x
      x3 = x * x2
  in x3 - fromInteger 5 * x2 + fromInteger 7 * x - fromInteger 3

-- Backward mode
public export
example4_back : Expr -> Expr
example4_back x = 
  SubExpr 
    (SubExpr 
      (AddExpr 
        (MulExpr (MulExpr x x) x) 
        (MulExpr (ConstExpr 7.0) x))
      (MulExpr (ConstExpr 5.0) (MulExpr x x)))
    (ConstExpr 3.0)

-- Example 5: f(x) = sin(x^2)
-- Forward mode
public export
example5_fwd : Dual -> Dual
example5_fwd x = sin (square x)

-- Backward mode
public export
example5_back : Expr -> Expr
example5_back x = SinExpr (MulExpr x x)

-- Example 6: f(x) = log(x) / x
-- Forward mode
public export
example6_fwd : Dual -> Dual
example6_fwd x = log x / x

-- Backward mode
public export
example6_back : Expr -> Expr
example6_back x = DivExpr (LogExpr x) x

-- Example 7: f(x) = cos(sin(x))
-- Forward mode
public export
example7_fwd : Dual -> Dual
example7_fwd x = cos (sin x)

-- Backward mode
public export
example7_back : Expr -> Expr
example7_back x = CosExpr (SinExpr x)

-- Example 8: f(x) = exp(sin(x) + cos(x))
-- Forward mode
public export
example8_fwd : Dual -> Dual
example8_fwd x = exp (sin x + cos x)

-- Backward mode
public export
example8_back : Expr -> Expr
example8_back x = ExpExpr (AddExpr (SinExpr x) (CosExpr x))

-- Example 9: f(x) = x / (1 + x^2)^2
-- Forward mode
public export
example9_fwd : Dual -> Dual
example9_fwd x = 
  let denominator = fromInteger 1 + square x
      denominator_squared = square denominator
  in x / denominator_squared

-- Backward mode
public export
example9_back : Expr -> Expr
example9_back x = 
  let denominator = PowExpr (AddExpr (ConstExpr 1.0) (MulExpr x x)) 2.0
  in DivExpr x denominator

-- Example 10: f(x) = x * exp(-x^2 / 2)
-- Forward mode
public export
example10_fwd : Dual -> Dual
example10_fwd x = 
  let exponent = (fromInteger (-1) * square x) / fromInteger 2
  in x * exp exponent

-- Backward mode
public export
example10_back : Expr -> Expr
example10_back x = 
  let exp_term = ExpExpr (DivExpr (MulExpr (ConstExpr (-1.0)) (MulExpr x x)) (ConstExpr 2.0))
  in MulExpr x exp_term

-- Run examples and print results
public export
run_examples : IO ()
run_examples = do
  -- Example 1: f(x) = x^2 + 3x + 2
  let x1 = 2.0
  
  putStrLn "Example 1: f(x) = x^2 + 3x + 2 at x = 2"
  putStrLn $ "  Value: " ++ show (eval (example1_back (makeVar x1)))
  putStrLn $ "  Forward Derivative: " ++ show (forward_derivative example1_fwd x1)
  putStrLn $ "  Backward Derivative: " ++ show (backward_derivative example1_back x1 1)
  putStrLn $ "  Second Derivative (Backward): " ++ show (backward_derivative example1_back x1 2)
  putStrLn $ "  Third Derivative (Backward): " ++ show (backward_derivative example1_back x1 3)
  putStrLn ""
  
  -- Example 2: f(x) = sin(x) * cos(x)
  let x2 = 1.0
  
  putStrLn "Example 2: f(x) = sin(x) * cos(x) at x = 1"
  putStrLn $ "  Value: " ++ show (eval (example2_back (makeVar x2)))
  putStrLn $ "  Forward Derivative: " ++ show (forward_derivative example2_fwd x2)
  putStrLn $ "  Backward Derivative: " ++ show (backward_derivative example2_back x2 1)
  putStrLn $ "  Second Derivative (Backward): " ++ show (backward_derivative example2_back x2 2)
  putStrLn $ "  Third Derivative (Backward): " ++ show (backward_derivative example2_back x2 3)
  putStrLn ""
  
  -- Example 3: f(x) = e^(x^2) / (1 + x^2)
  let x3 = 1.5
  
  putStrLn "Example 3: f(x) = e^(x^2) / (1 + x^2) at x = 1.5"
  putStrLn $ "  Value: " ++ show (eval (example3_back (makeVar x3)))
  putStrLn $ "  Forward Derivative: " ++ show (forward_derivative example3_fwd x3)
  putStrLn $ "  Backward Derivative: " ++ show (backward_derivative example3_back x3 1)
  putStrLn $ "  Second Derivative (Backward): " ++ show (backward_derivative example3_back x3 2)
  putStrLn $ "  Third Derivative (Backward): " ++ show (backward_derivative example3_back x3 3)
  putStrLn ""
  
  -- Example 4: f(x) = x^3 - 5x^2 + 7x - 3
  let x4 = 2.0
  
  putStrLn "Example 4: f(x) = x^3 - 5x^2 + 7x - 3 at x = 2"
  putStrLn $ "  Value: " ++ show (eval (example4_back (makeVar x4)))
  putStrLn $ "  Forward Derivative: " ++ show (forward_derivative example4_fwd x4)
  putStrLn $ "  Backward Derivative: " ++ show (backward_derivative example4_back x4 1)
  putStrLn $ "  Second Derivative (Backward): " ++ show (backward_derivative example4_back x4 2)
  putStrLn $ "  Third Derivative (Backward): " ++ show (backward_derivative example4_back x4 3)
  putStrLn ""
  
  -- Example 5: f(x) = sin(x^2)
  let x5 = 1.5
  
  putStrLn "Example 5: f(x) = sin(x^2) at x = 1.5"
  putStrLn $ "  Value: " ++ show (eval (example5_back (makeVar x5)))
  putStrLn $ "  Forward Derivative: " ++ show (forward_derivative example5_fwd x5)
  putStrLn $ "  Backward Derivative: " ++ show (backward_derivative example5_back x5 1)
  putStrLn $ "  Second Derivative (Backward): " ++ show (backward_derivative example5_back x5 2)
  putStrLn $ "  Third Derivative (Backward): " ++ show (backward_derivative example5_back x5 3)
  putStrLn ""
  
  -- Example 6: f(x) = log(x) / x
  let x6 = 3.0
  
  putStrLn "Example 6: f(x) = log(x) / x at x = 3"
  putStrLn $ "  Value: " ++ show (eval (example6_back (makeVar x6)))
  putStrLn $ "  Forward Derivative: " ++ show (forward_derivative example6_fwd x6)
  putStrLn $ "  Backward Derivative: " ++ show (backward_derivative example6_back x6 1)
  putStrLn $ "  Second Derivative (Backward): " ++ show (backward_derivative example6_back x6 2)
  putStrLn $ "  Third Derivative (Backward): " ++ show (backward_derivative example6_back x6 3)
  putStrLn ""
  
  -- Example 7: f(x) = cos(sin(x))
  let x7 = 0.5
  
  putStrLn "Example 7: f(x) = cos(sin(x)) at x = 0.5"
  putStrLn $ "  Value: " ++ show (eval (example7_back (makeVar x7)))
  putStrLn $ "  Forward Derivative: " ++ show (forward_derivative example7_fwd x7)
  putStrLn $ "  Backward Derivative: " ++ show (backward_derivative example7_back x7 1)
  putStrLn $ "  Second Derivative (Backward): " ++ show (backward_derivative example7_back x7 2)
  putStrLn $ "  Third Derivative (Backward): " ++ show (backward_derivative example7_back x7 3)
  putStrLn ""
  
  -- Example 8: f(x) = exp(sin(x) + cos(x))
  let x8 = 1.0
  
  putStrLn "Example 8: f(x) = exp(sin(x) + cos(x)) at x = 1"
  putStrLn $ "  Value: " ++ show (eval (example8_back (makeVar x8)))
  putStrLn $ "  Forward Derivative: " ++ show (forward_derivative example8_fwd x8)
  putStrLn $ "  Backward Derivative: " ++ show (backward_derivative example8_back x8 1)
  putStrLn $ "  Second Derivative (Backward): " ++ show (backward_derivative example8_back x8 2)
  putStrLn $ "  Third Derivative (Backward): " ++ show (backward_derivative example8_back x8 3)
  putStrLn ""
  
  -- Example 9: f(x) = x / (1 + x^2)^2
  let x9 = 2.0
  
  putStrLn "Example 9: f(x) = x / (1 + x^2)^2 at x = 2"
  putStrLn $ "  Value: " ++ show (eval (example9_back (makeVar x9)))
  putStrLn $ "  Forward Derivative: " ++ show (forward_derivative example9_fwd x9)
  putStrLn $ "  Backward Derivative: " ++ show (backward_derivative example9_back x9 1)
  putStrLn $ "  Second Derivative (Backward): " ++ show (backward_derivative example9_back x9 2)
  putStrLn $ "  Third Derivative (Backward): " ++ show (backward_derivative example9_back x9 3)
  putStrLn ""
  
  -- Example 10: f(x) = x * exp(-x^2 / 2)
  let x10 = 1.0
  
  putStrLn "Example 10: f(x) = x * exp(-x^2 / 2) at x = 1"
  putStrLn $ "  Value: " ++ show (eval (example10_back (makeVar x10)))
  putStrLn $ "  Forward Derivative: " ++ show (forward_derivative example10_fwd x10)
  putStrLn $ "  Backward Derivative: " ++ show (backward_derivative example10_back x10 1)
  putStrLn $ "  Second Derivative (Backward): " ++ show (backward_derivative example10_back x10 2)
  putStrLn $ "  Third Derivative (Backward): " ++ show (backward_derivative example10_back x10 3)

-- Main function
public export
main : IO ()
main = run_examples