//initialize product to zero
@R2
M=0

//put index R1 in R3
@R1
D=M
@R3
M=D

(LOOP)
//check if loop ended. if yes, then jump
@R3
D=M
@END
D;JEQ

//loop not ended
//calculation
@R0
D=M
@R2
M=M+D

//decrement counter
@R3
M=M-1

@LOOP
0;JMP

(END)
@END
0;JMP
