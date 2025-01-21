# perl tokenizer.pl -l en -time < /home/user15/RNN/dataset2/train.en > /home/user15/RNN/dataset3/train.en
# #perl tokenizer.pl -l en -time -a -no-escape < /home/user15/RNN/dataset/dev/newstest_2013.en > /home/user15/RNN/dataset3/newstest_2013.en
# #perl tokenizer.pl -l en -time -a -no-escape < /home/user15/RNN/dataset2/newstest2014.en > /home/user15/RNN/dataset3/newstest2014.en

# perl tokenizer.pl -l de -time  < /home/user15/RNN/dataset2/train.de > /home/user15/RNN/dataset3/train.de
# #perl tokenizer.pl -l de -time -a -no-escape < /home/user15/RNN/dataset/dev/newstest_2013.de > /home/user15/RNN/dataset3/newstest_2013.de
# #perl tokenizer.pl -l de -time -a -no-escape < /home/user15/RNN/dataset2/newstest2014.de > /home/user15/RNN/dataset3/newstest2014.de


cat /home/user15/RNN/dataset2/newstest2014.de | ./cus_tokenizer.pl > /home/user15/RNN/dataset5/newstest2014.de

cat /home/user15/RNN/dataset2/newstest2014.en | ./cus_tokenizer.pl > /home/user15/RNN/dataset5/newstest2014.en

cat /home/user15/RNN/dataset2/train.de | ./cus_tokenizer.pl > /home/user15/RNN/dataset5/train.de

cat /home/user15/RNN/dataset2/train.en | ./cus_tokenizer.pl > /home/user15/RNN/dataset5/train.en


cat /home/user15/RNN/dataset/dev/newstest_2013.de | ./cus_tokenizer.pl > /home/user15/RNN/dataset5/newstest_2013.de

cat /home/user15/RNN/dataset/dev/newstest_2013.en | ./cus_tokenizer.pl > /home/user15/RNN/dataset5/newstest_2013.en
