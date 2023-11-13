import random
import re

qa_dict = {
    "대학원 면접은 어떻게 준비해야 하나요?": ["면접 연습", "자기소개서 작성", "학문적 역량 강화"],
    "본인의 연구 관심사는 무엇인가요?": ["연구 주제", "연구 분야", "연구 관심사"],
    # 다른 질문들과 그에 대한 키워드를 추가
}

not_used_yet = list(qa_dict.keys())

def get_random_question():
    random_question = random.choice(not_used_yet)
    not_used_yet.remove(random_question) 
    return random_question

def main():
    # 입력값을 리스트로 저장하고, 리스트에 저장한 값과 qa_dict 의 value 값들을 비교하면 될 것 같음 강등펀치   
    while (not_used_yet):
        question = get_random_question()
        print(f"질문: {question}")
        user_answer = input("답변을 입력하세요: ")
        user_answer = re.sub(r'[-,\s]',user_answer)
        ans = False
        while ans == False:
            if all(keyword in user_answer for keyword in qa_dict[question]):
                ans = True
                print('Correct!')
            else:
                # 여기 부분을 조금 수정
                ans = False
                print("다시 해보세요. 정답을 보시려면, '정답'이라고 입력해주세요.") 
                print(f"질문: {question}")
                retry = input("(계속/정답):")
                if retry == '정답':
                    print(f'정답 키워드: {qa_dict[question]}')
                    break

        next_question = input("다음 질문을 받아보시겠습니까? (y/n): ")
        if (next_question.lower() == 'y'):
            print(f'남은 문제 수: {len(not_used_yet)}')
        else:
            print(f'수고하셨습니다. 프로그램을 종료합니다.')
            break

if __name__ == "__main__":
    main()