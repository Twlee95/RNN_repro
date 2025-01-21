#!/usr/bin/perl -w

use strict;
use utf8;
binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");

# 기본 문자열 처리 함수
sub basic_tokenizer {
    my ($text) = @_;

    $text =~ s/[\000-\037]//g;

    # HTML escape for single quotes
    $text =~ s/[\'’‘ʻʼ՚]/&apos;/g;

    $text =~ s/&apos;\s*t\b/ &apos;t /g;
    $text =~ s/&apos;\s*m\b/ &apos;m /g;
    $text =~ s/&apos;\s*re\b/ &apos;re /g;
    $text =~ s/&apos;\s*s\b/ &apos;s /g;
    $text =~ s/&apos;\s*d\b/ &apos;d /g;
    $text =~ s/&apos;\s*ll\b/ &apos;ll /g;
    $text =~ s/&apos;\s*ve\b/ &apos;ve /g;

    $text =~ s/&apos;([sSmMdD])/&apos;$1 /g;
    $text =~ s/&apos;ll/&apos;ll /g;
    $text =~ s/&apos;re/&apos;re /g;
    $text =~ s/&apos;ve/&apos;ve /g;
    $text =~ s/&apos;t/&apos;t /g;
    $text =~ s/&apos;LL/&apos;LL /g;
    $text =~ s/&apos;RE/&apos;RE /g;
    $text =~ s/&apos;VE/&apos;VE /g;
    $text =~ s/&apos;T/&apos;T /g;

    $text =~ s/\x{2026}/ MULTIDOT /g; # …
    $text =~ s/\.\.+/ MULTIDOT /g;


    # HTML escape for double quotes
    $text =~ s/[\x22„“”«»‟]/ &quot; /g;

    # 쉼표로 구분된 숫자 (예: 1,234 -> NUM)
    $text =~ s/\b\d+(?:,\d+)+\b/NUM/g;

    # 소수 포함 숫자 (예: 3.14 -> NUM)
    $text =~ s/\b\d+(?:\.\d+)?\b/NUM/g;

    # 과학적 표기법 숫자 (예: 1.23e10 -> NUM)
    $text =~ s/\b\d+\.\d+[eE][+-]?\d+\b/NUM/g;

    # 서수 표현 (예: 21st, 150th -> NUM)
    $text =~ s/\b\d+(?:st|nd|rd|th)\b/NUM/g;

    # 시간 표현 (예: 11pm, 2.15am -> NUM)
    $text =~ s/\b\d+(?:\.\d+)?[apAP][mM]\b/NUM/g;

    # 연대 표현 (예: 1970s -> NUM)
    $text =~ s/\b\d{2,}s\b/NUM/g;

    # 단위 표현 (예: 23km, 3m, 5kg -> NUM)
    $text =~ s/\b\d+(?:\.\d+)?(?:km|m|cm|mm|kg|g|mg|l|ml|lb|oz|ft|yd|mi|in|h|sec|min|hr)\b/NUM/g;

    # $text =~ s/(?<!\s)(&apos;)/ $1/g;
    # $text =~ s/(?<!\s)(&quot;)/ $1/g;

    # # 뒤에 공백이 없으면 추가
    # $text =~ s/(&apos;)(?!\s)/$1 /g;
    # $text =~ s/(&quot;)(?!\s)/$1 /g;

    # HTML escape for less-than and greater-than
    # $text =~ s/</&lt;/g;
    # $text =~ s/>/&gt;/g;

    $text =~ s/�//g;
    # Step 3: 사용자 정의 처리 ’

    $text =~ s/\s+/ /g;         # 연속된 공백을 단일 공백으로 치환
    $text =~ s/^[\t ]+|[\t ]+$//g;
    
    return $text;
}

# 표준 입력으로부터 문자열을 읽어와 처리 후 출력
while (<STDIN>) {
    chomp;
    my $tokenized_text = basic_tokenizer($_);
    print "$tokenized_text\n";
}