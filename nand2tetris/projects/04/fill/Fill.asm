(LOOP)
//Store index i for looping in R0
@8192
D=A
@R0
M=D

//Starting point of screen
@16384
D=A
@R1
M=D

//Check if we need to black out screen or white it out
@24576
D=M
@WHITE
D;JEQ


(BLACK)
//Black out the pixel pointed by R1
@R1
A=M
M=-1
//update value of R1 to point to next pixel
A=A+1
D=A
@R1
M=D
//decrement index i in R0 
@R0
M=M-1
D=M
@BLACK
D;JNE

@LOOP
0;JMP

(WHITE)
//White out the pixel pointed by R1
@R1
A=M
M=0
//update value of R1 to point to next pixel
A=A+1
D=A
@R1
M=D
//decrement index i in R0 
@R0
M=M-1
D=M
@WHITE
D;JNE

@LOOP
0;JMP
