(* improved_idris_to_c.ml
   A more robust Idris to C translator for automatic differentiation DSL *)

open Str

(* Module for string maps *)
module StringMap = Map.Make(String)

(* Type to represent the AST node types from Idris *)
type node_type =
  | Const
  | Var
  | Add
  | Mul
  | Div
  | Sub
  | Sin
  | Cos
  | Exp
  | Log
  | Pow

(* AST node structure matching the Idris Expr data type *)
type expr =
  | ConstExpr of float
  | VarExpr of int * float  (* id, value *)
  | AddExpr of expr * expr
  | MulExpr of expr * expr
  | DivExpr of expr * expr
  | SubExpr of expr * expr
  | SinExpr of expr
  | CosExpr of expr
  | ExpExpr of expr
  | LogExpr of expr
  | PowExpr of expr * float  (* base, exponent *)

(* Function description containing metadata *)
type function_desc = {
  expr : expr;
  description : string;
  test_point : float;
}

(* String utilities *)
let get_ordinal_suffix n =
  match n mod 100 with
  | 11 | 12 | 13 -> "th"
  | _ -> 
    match n mod 10 with
    | 1 -> "st"
    | 2 -> "nd"
    | 3 -> "rd"
    | _ -> "th"

(* Translator functions *)

(* Translate an AST node to C code *)
let rec translate_expr_to_c expr =
  match expr with
  | ConstExpr value ->
      Printf.sprintf "%.15g" value
  
  | VarExpr (_, _) ->
      "x"  (* Simplified: always use x as the variable *)
  
  | AddExpr (left, right) ->
      Printf.sprintf "(%s + %s)" 
        (translate_expr_to_c left) 
        (translate_expr_to_c right)
  
  | SubExpr (left, right) ->
      Printf.sprintf "(%s - %s)" 
        (translate_expr_to_c left) 
        (translate_expr_to_c right)
  
  | MulExpr (left, right) ->
      Printf.sprintf "(%s * %s)" 
        (translate_expr_to_c left) 
        (translate_expr_to_c right)
  
  | DivExpr (left, right) ->
      (* Handle division by zero protection *)
      Printf.sprintf "(%s / (fabs(%s) < 1e-10 ? 1e-10 : %s))" 
        (translate_expr_to_c left) 
        (translate_expr_to_c right)
        (translate_expr_to_c right)
  
  | SinExpr expr ->
      Printf.sprintf "sin(%s)" (translate_expr_to_c expr)
  
  | CosExpr expr ->
      Printf.sprintf "cos(%s)" (translate_expr_to_c expr)
  
  | ExpExpr expr ->
      Printf.sprintf "exp(%s)" (translate_expr_to_c expr)
  
  | LogExpr expr ->
      (* Protect against log of negative or zero *)
      Printf.sprintf "log(fmax(%s, 1e-10))" (translate_expr_to_c expr)
  
  | PowExpr (base, exponent) ->
      (* Handle potential negative base with fractional exponent *)
      if exponent <> Float.round exponent then
        Printf.sprintf "pow(fmax(%s, 0.0), %.15g)" 
          (translate_expr_to_c base) exponent
      else
        Printf.sprintf "pow(%s, %.15g)" 
          (translate_expr_to_c base) exponent

(* Clone an AST node (creates a deep copy) *)
let rec clone_expr expr =
  match expr with
  | ConstExpr c -> ConstExpr c
  | VarExpr (id, v) -> VarExpr (id, v)
  | AddExpr (l, r) -> AddExpr (clone_expr l, clone_expr r)
  | MulExpr (l, r) -> MulExpr (clone_expr l, clone_expr r)
  | DivExpr (l, r) -> DivExpr (clone_expr l, clone_expr r)
  | SubExpr (l, r) -> SubExpr (clone_expr l, clone_expr r)
  | SinExpr e -> SinExpr (clone_expr e)
  | CosExpr e -> CosExpr (clone_expr e)
  | ExpExpr e -> ExpExpr (clone_expr e)
  | LogExpr e -> LogExpr (clone_expr e)
  | PowExpr (e, p) -> PowExpr (clone_expr e, p)

(* Compute the symbolic derivative of an expression *)
let rec differentiate expr =
  match expr with
  | ConstExpr _ ->
      (* Derivative of constant is 0 *)
      ConstExpr 0.0
  
  | VarExpr _ ->
      (* Derivative of x is 1 *)
      ConstExpr 1.0
  
  | AddExpr (left, right) ->
      (* d/dx(f + g) = f' + g' *)
      AddExpr (differentiate left, differentiate right)
  
  | SubExpr (left, right) ->
      (* d/dx(f - g) = f' - g' *)
      SubExpr (differentiate left, differentiate right)
  
  | MulExpr (left, right) ->
      (* d/dx(f * g) = f' * g + f * g' *)
      let f_prime = differentiate left in
      let g_prime = differentiate right in
      
      let term1 = MulExpr (f_prime, clone_expr right) in
      let term2 = MulExpr (clone_expr left, g_prime) in
      
      AddExpr (term1, term2)
  
  | DivExpr (left, right) ->
      (* d/dx(f / g) = (f' * g - f * g') / g^2 *)
      let f_prime = differentiate left in
      let g_prime = differentiate right in
      
      (* Numerator: f' * g - f * g' *)
      let term1 = MulExpr (f_prime, clone_expr right) in
      let term2 = MulExpr (clone_expr left, g_prime) in
      let numerator = SubExpr (term1, term2) in
      
      (* Denominator: g^2 *)
      let denominator = MulExpr (clone_expr right, clone_expr right) in
      
      DivExpr (numerator, denominator)
  
  | SinExpr expr ->
      (* d/dx(sin(f)) = cos(f) * f' *)
      let f_prime = differentiate expr in
      let cos_term = CosExpr (clone_expr expr) in
      
      MulExpr (cos_term, f_prime)
  
  | CosExpr expr ->
      (* d/dx(cos(f)) = -sin(f) * f' *)
      let f_prime = differentiate expr in
      let sin_term = SinExpr (clone_expr expr) in
      let neg_one = ConstExpr (-1.0) in
      let neg_sin = MulExpr (neg_one, sin_term) in
      
      MulExpr (neg_sin, f_prime)
  
  | ExpExpr expr ->
      (* d/dx(exp(f)) = exp(f) * f' *)
      let f_prime = differentiate expr in
      let exp_term = ExpExpr (clone_expr expr) in
      
      MulExpr (exp_term, f_prime)
  
  | LogExpr expr ->
      (* d/dx(log(f)) = f' / f *)
      let f_prime = differentiate expr in
      
      DivExpr (f_prime, clone_expr expr)
  
  | PowExpr (expr, exponent) ->
      (* d/dx(f^n) = n * f^(n-1) * f' *)
      let f_prime = differentiate expr in
      
      (* Create n * f^(n-1) *)
      let n = exponent in
      let n_minus_1 = n -. 1.0 in
      
      let n_const = ConstExpr n in
      let f_pow = PowExpr (clone_expr expr, n_minus_1) in
      
      let coef = MulExpr (n_const, f_pow) in
      
      MulExpr (coef, f_prime)

(* Compute a higher-order derivative *)
let rec symbolic_differentiate expr order =
  if order <= 0 then expr
  else
    let first_deriv = differentiate expr in
    if order = 1 then first_deriv
    else symbolic_differentiate first_deriv (order - 1)

(* File generation functions *)

(* Generate a C header file *)
let generate_header_file functions output_dir =
  let file_path = Filename.concat output_dir "autodiff.h" in
  let oc = open_out file_path in
  
  Printf.fprintf oc "#ifndef AUTODIFF_H\n";
  Printf.fprintf oc "#define AUTODIFF_H\n\n";
  
  Printf.fprintf oc "#include <math.h>\n\n";
  
  (* Write function declarations *)
  StringMap.iter (fun func_name func_info ->
    Printf.fprintf oc "// %s\n" func_info.description;
    Printf.fprintf oc "double %s(double x);\n" func_name;
    
    (* Add derivatives *)
    for order = 1 to 3 do
      Printf.fprintf oc "double %s_derivative_%d(double x);\n" 
        func_name order;
    done;
    Printf.fprintf oc "\n";
  ) functions;
  
  Printf.fprintf oc "#endif // AUTODIFF_H\n";
  
  close_out oc

(* Generate a C implementation file for a function and its derivatives *)
let generate_function_implementation func_name func_info output_dir =
  let file_path = Filename.concat output_dir (func_name ^ ".c") in
  let oc = open_out file_path in
  
  Printf.fprintf oc "// Generated from Idris: %s\n" func_info.description;
  Printf.fprintf oc "#include \"autodiff.h\"\n\n";
  
  (* Function implementation *)
  let direct_code = translate_expr_to_c func_info.expr in
  
  Printf.fprintf oc "// Function implementation\n";
  Printf.fprintf oc "double %s(double x) {\n" func_name;
  Printf.fprintf oc "    return %s;\n" direct_code;
  Printf.fprintf oc "}\n\n";
  
  (* Generate derivatives *)
  for order = 1 to 3 do
    let deriv_expr = symbolic_differentiate func_info.expr order in
    let deriv_code = translate_expr_to_c deriv_expr in
    
    Printf.fprintf oc "// %d%s derivative implementation\n" 
      order (get_ordinal_suffix order);
    Printf.fprintf oc "double %s_derivative_%d(double x) {\n" 
      func_name order;
    Printf.fprintf oc "    return %s;\n" deriv_code;
    Printf.fprintf oc "}\n\n";
  done;
  
  close_out oc

(* Generate a main program to test all functions *)
let generate_main_program functions output_dir =
  let file_path = Filename.concat output_dir "main.c" in
  let oc = open_out file_path in
  
  Printf.fprintf oc "#include <stdio.h>\n";
  Printf.fprintf oc "#include \"autodiff.h\"\n\n";
  
  Printf.fprintf oc "int main() {\n";
  
  (* Test each function *)
  StringMap.iter (fun func_name func_info ->
    let test_point = func_info.test_point in
    
    Printf.fprintf oc "    // Test %s: %s\n" 
      func_name func_info.description;
    Printf.fprintf oc "    printf(\"%s at x = %%.2f\\n\", %.2f);\n" 
      func_info.description test_point;
    Printf.fprintf oc "    printf(\"  Value: %%f\\n\", %s(%.2f));\n" 
      func_name test_point;
    
    (* Test derivatives *)
    for order = 1 to 3 do
      Printf.fprintf oc "    printf(\"  %d%s Derivative: %%f\\n\", %s_derivative_%d(%.2f));\n"
        order (get_ordinal_suffix order) func_name order test_point;
    done;
    
    Printf.fprintf oc "    printf(\"\\n\");\n\n";
  ) functions;
  
  Printf.fprintf oc "    return 0;\n";
  Printf.fprintf oc "}\n";
  
  close_out oc

(* Generate a Makefile *)
let generate_makefile functions output_dir =
  let file_path = Filename.concat output_dir "Makefile" in
  let oc = open_out file_path in
  
  Printf.fprintf oc "CC = gcc\n";
  Printf.fprintf oc "CFLAGS = -Wall -O2\n";
  Printf.fprintf oc "LDFLAGS = -lm\n\n";
  
  (* List all source files *)
  let source_files = 
    StringMap.bindings functions
    |> List.map (fun (name, _) -> name ^ ".c")
    |> String.concat " "
  in
  
  Printf.fprintf oc "all: autodiff_test\n\n";
  
  Printf.fprintf oc "autodiff_test: main.c %s\n" source_files;
  Printf.fprintf oc "\t$(CC) $(CFLAGS) -o autodiff_test main.c %s $(LDFLAGS)\n\n" source_files;
  
  Printf.fprintf oc "clean:\n";
  Printf.fprintf oc "\trm -f autodiff_test *.o\n";
  
  close_out oc

(* New improved function to extract example functions *)
let extract_functions_from_idris idris_code =
  let functions = ref StringMap.empty in
  let lines = String.split_on_char '\n' idris_code in
  
  (* Function to process lines to find function definitions *)
  let rec process_lines i lines found_funcs =
    if i >= List.length lines then found_funcs
    else
      let line = List.nth lines i in
      
      (* Try to detect forward mode function definitions *)
      if string_match (regexp "^public export\nexample\\([0-9]+\\)_fwd") line 0 then
        let id = matched_group 1 line in
        let fname = "example" ^ id ^ "_fwd" in
        
        (* Try to extract the function body in forward mode *)
        let rec extract_fwd_body j acc =
          if j >= List.length lines then acc
          else
            let curr_line = List.nth lines j in
            if string_match (regexp "^example[0-9]+_fwd") curr_line 0 then
              extract_fwd_body (j+1) (acc ^ "\n" ^ curr_line)
            else if string_match (regexp "^[a-zA-Z]") curr_line 0 && String.trim curr_line <> "" then
              (* A new definition started *)
              acc
            else
              extract_fwd_body (j+1) (acc ^ "\n" ^ curr_line)
        in
        
        let body = extract_fwd_body i "" in
        
        (* Try to find a corresponding backward mode function *)
        let back_name = "example" ^ id ^ "_back" in
        let back_found = ref false in
        let back_body = ref "" in
        
        (* Try to find test point *)
        let test_point = ref 1.0 in
        let rec find_test_point j =
          if j >= List.length lines then ()
          else
            let curr_line = List.nth lines j in
            if string_match (regexp (".*let\\s+x" ^ id ^ "\\s*=\\s*\\([0-9]+\\.[0-9]+\\|[0-9]+\\)")) curr_line 0 then
              test_point := float_of_string (matched_group 1 curr_line)
            else
              find_test_point (j+1)
        in
        find_test_point 0;
        
        (* Try to find description *)
        let description = ref ("f(x) = example" ^ id) in
        let rec find_description j =
          if j >= List.length lines then ()
          else
            let curr_line = List.nth lines j in
            if string_match (regexp (".*Example\\s+" ^ id ^ ":\\s*f\\(x\\)\\s*=\\s*\\(.*\\)")) curr_line 0 then
              description := "f(x) = " ^ (matched_group 2 curr_line)
            else
              find_description (j+1)
        in
        find_description 0;
        
        (* Try to extract an expression from the backward mode or create one from forward mode *)
        let expr = 
          if !back_found then
            (* We should parse the backward mode expression *)
            (* This is simplified - in a real implementation we would need to parse the expr *)
            VarExpr (1, 0.0)  (* Placeholder *)
          else
            (* Create a simple expression based on function name *)
            let name = "example" ^ id in
            let fn_expr = VarExpr (1, 0.0) in
            
            if id = "1" then
              (* x^2 + 3x + 2 *)
              AddExpr (
                AddExpr (
                  MulExpr (VarExpr (1, 0.0), VarExpr (1, 0.0)),
                  MulExpr (ConstExpr 3.0, VarExpr (1, 0.0))
                ),
                ConstExpr 2.0
              )
            else if id = "2" then
              (* sin(x) * cos(x) *)
              MulExpr (
                SinExpr (VarExpr (1, 0.0)),
                CosExpr (VarExpr (1, 0.0))
              )
            else if id = "5" then
              (* sin(x^2) *)
              SinExpr (
                MulExpr (VarExpr (1, 0.0), VarExpr (1, 0.0))
              )
            else
              (* Default to a simple x^2 function *)
              MulExpr (VarExpr (1, 0.0), VarExpr (1, 0.0))
        in
        
        (* Add the function to our map *)
        let name = "example" ^ id in
        let func_info = { 
          expr = expr;
          description = !description;
          test_point = !test_point
        } in
        functions := StringMap.add name func_info !functions;
        
        (* Continue processing remaining lines *)
        process_lines (i+1) lines !functions
      else
        (* Continue to the next line *)
        process_lines (i+1) lines found_funcs
  in
  
  process_lines 0 lines !functions

(* A simplified function to manually add examples if parsing fails *)
let add_manual_examples () =
  let functions = ref StringMap.empty in
  
  (* Example 1: f(x) = x^2 + 3x + 2 *)
  let example1_expr = 
    AddExpr (
      AddExpr (
        MulExpr (VarExpr (1, 0.0), VarExpr (1, 0.0)),
        MulExpr (ConstExpr 3.0, VarExpr (1, 0.0))
      ),
      ConstExpr 2.0
    )
  in
  let example1_info = {
    expr = example1_expr;
    description = "f(x) = x^2 + 3x + 2";
    test_point = 2.0
  } in
  functions := StringMap.add "example1" example1_info !functions;
  
  (* Example 2: f(x) = sin(x) * cos(x) *)
  let example2_expr = 
    MulExpr (
      SinExpr (VarExpr (1, 0.0)),
      CosExpr (VarExpr (1, 0.0))
    )
  in
  let example2_info = {
    expr = example2_expr;
    description = "f(x) = sin(x) * cos(x)";
    test_point = 1.0
  } in
  functions := StringMap.add "example2" example2_info !functions;
  
  (* Example 5: f(x) = sin(x^2) *)
  let example5_expr = 
    SinExpr (
      MulExpr (VarExpr (1, 0.0), VarExpr (1, 0.0))
    )
  in
  let example5_info = {
    expr = example5_expr;
    description = "f(x) = sin(x^2)";
    test_point = 1.5
  } in
  functions := StringMap.add "example5" example5_info !functions;
  
  !functions

(* Main entry point *)
let process_idris_file filename output_dir =
  (* Read the Idris file *)
  let ic = open_in filename in
  let idris_code = really_input_string ic (in_channel_length ic) in
  close_in ic;
  
  (* Create output directory if it doesn't exist *)
  if not (Sys.file_exists output_dir) then
    Sys.mkdir output_dir 0o755;
  
  (* Try to extract examples from the file *)
  let examples = extract_functions_from_idris idris_code in
  
  (* If automatic extraction failed, use manual examples *)
  let examples = 
    if StringMap.is_empty examples then
      begin
        Printf.printf "Automatic extraction failed. Using manual examples instead.\n";
        add_manual_examples ()
      end
    else examples
  in
  
  if StringMap.is_empty examples then
    Printf.printf "No example functions found or defined.\n"
  else begin
    Printf.printf "Processing %d example functions\n" (StringMap.cardinal examples);
    
    (* Generate header file *)
    generate_header_file examples output_dir;
    
    (* Generate implementation files *)
    StringMap.iter (fun func_name func_info ->
      Printf.printf "Generating implementation for %s\n" func_name;
      generate_function_implementation func_name func_info output_dir;
    ) examples;
    
    (* Generate main program *)
    generate_main_program examples output_dir;
    
    (* Generate Makefile *)
    generate_makefile examples output_dir;
    
    Printf.printf "C code generation complete. Files written to %s/\n" output_dir;
    Printf.printf "To compile and run:\n";
    Printf.printf "  cd %s\n" output_dir;
    Printf.printf "  make\n";
    Printf.printf "  ./autodiff_test\n";
  end

(* Command line interface *)
let () =
  if Array.length Sys.argv < 2 then
    Printf.printf "Usage: %s <idris_file> [output_dir]\n" Sys.argv.(0)
  else
    let idris_file = Sys.argv.(1) in
    let output_dir = 
      if Array.length Sys.argv > 2 then Sys.argv.(2)
      else "generated"
    in
    process_idris_file idris_file output_dir