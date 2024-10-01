# 开篇词 (1讲) 开篇词｜开发工程师如何进阶为AI应用型人才？
## 转型为 AI 应用型工程师的难点
![image](https://github.com/user-attachments/assets/94da3a81-a88c-49e3-b2c1-d967711c3729)

## 开发工程师入局 AI 的最佳路径
![image](https://github.com/user-attachments/assets/e2cdb21d-36ed-433a-8d32-e488d9686907)

## 这门课程是如何设计的？
根据这条最佳的学习路径，由浅入深地把课程分成了 5 个章节。

![image](https://github.com/user-attachments/assets/a8f77474-f072-4699-99c4-73e44c19bf6b)

# 第一章：小试牛刀，理解基础概念 (3讲) 01｜洞察本质：从工程学角度看ChatGPT为什么会崛起
ChatGPT 具体是如何赢得这场胜利的呢？我们一一来看。

## NLP 技术突破：强势整合技术资源
![image](https://github.com/user-attachments/assets/b53e236f-ea06-477a-92a3-7041c7cb47c2)

## 基于自回归的无监督训练
![image](https://github.com/user-attachments/assets/dba9fc35-117d-42e0-a170-6b6e11464c9b)

## 与人类意识对齐（Alignment）
![image](https://github.com/user-attachments/assets/b7c0808b-e5c5-4071-9b98-e57f007a50e4)

![image](https://github.com/user-attachments/assets/800caa4d-a5d7-4fe2-ad11-76009fbf412f)

![image](https://github.com/user-attachments/assets/f0312037-dd20-44fb-900a-fa1844925798)

## 突现能力（Emergent Ability）
![image](https://github.com/user-attachments/assets/457e0e37-76e6-4712-9277-88431d1e5c67)

![image](https://github.com/user-attachments/assets/b8d87364-159c-4b27-b3a2-a50bdf0fbd0a)

## 超大规模数据集：超过 40T 的文本数据
![image](https://github.com/user-attachments/assets/fbeb4fa4-9b34-4b86-81aa-caa616e9ce4d)

![image](https://github.com/user-attachments/assets/2e859aa0-d546-47e1-8788-59187f88a60c)

## 找对了金主爸爸
![image](https://github.com/user-attachments/assets/f1ba684b-3f2b-4dec-a244-233e560cbe33)

## 产品化开放：让大家随便玩
![image](https://github.com/user-attachments/assets/49617884-3b97-4901-b984-dbf9b2c76798)

## 便捷使用
![image](https://github.com/user-attachments/assets/b06b3906-f538-4589-9948-fd7936ef2080)

## 适用场景多
![image](https://github.com/user-attachments/assets/c224a392-c88f-4bad-960e-2ccc73571280)

## 使用效果好
![image](https://github.com/user-attachments/assets/f26ea80c-08bd-4113-a287-905c30eb54fb)

## 工程化应用
![image](https://github.com/user-attachments/assets/4273890d-934d-4fde-9f47-11e3b2425e96)

## 小结
![image](https://github.com/user-attachments/assets/a055a795-744a-4f72-bbd0-f5172f413ca1)

# 第一章：小试牛刀，理解基础概念 (3讲) 02｜学好提示工程，轻松驾驭大模型
**能否充分使用好 AI 大模型，提示是关键**。

## 什么是提示？
![image](https://github.com/user-attachments/assets/c6277e4d-a968-4c68-8687-f498c09f4bd3)

## 什么是提示工程？
![image](https://github.com/user-attachments/assets/d97ed688-f273-434d-95ae-e850c8cb4e9d)

## 什么是 AI 领导力？
![image](https://github.com/user-attachments/assets/1d15492c-63f2-4f68-991a-949f3b3490ce)

## 如何构造好的提示？
![image](https://github.com/user-attachments/assets/d6fef219-4c20-4133-aae4-8f824a8a87bc)

![image](https://github.com/user-attachments/assets/4ed235b8-2d62-46d6-9391-f569fb4963e0)

```
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class LoginModule {
    private Map<String, String> userDatabase;

    public LoginModule() {
        // 初始化用户数据库，实际应用中通常会连接到数据库
        userDatabase = new HashMap<>();
        userDatabase.put("user1", "password1");
        userDatabase.put("user2", "password2");
        // 添加更多用户...
    }

    public boolean authenticateUser(String username, String password) {
        // 在实际应用中，这里通常会连接到数据库，验证用户名和密码是否匹配
        String storedPassword = userDatabase.get(username);
        return storedPassword != null && storedPassword.equals(password);
    }

    public static void main(String[] args) {
        LoginModule loginModule = new LoginModule();
        Scanner scanner = new Scanner(System.in);

        System.out.print("请输入用户名: ");
        String username = scanner.nextLine();

        System.out.print("请输入密码: ");
        String password = scanner.nextLine();

        if (loginModule.authenticateUser(username, password)) {
            System.out.println("登录成功！");
        } else {
            System.out.println("登录失败，用户名或密码错误！");
        }

        scanner.close();
    }
}

```

![image](https://github.com/user-attachments/assets/b24af86c-ef54-495b-8070-fc4b39ad017d)

```
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class UserController {

    @PostMapping("/login")
    public String login(@RequestBody UserLoginRequest request) {
        // 第二步：非空验证
        if (request.getUsername() == null || request.getPassword() == null) {
            return "用户名和密码不能为空";
        }

        // 第三步：数据库验证
        if (isValidUser(request.getUsername(), request.getPassword())) {
            return "登录成功";
        } else {
            return "用户名或密码错误";
        }
    }

    private boolean isValidUser(String username, String password) {
        // 这里应该调用数据库查询用户信息的方法，进行用户名和密码的验证
        // 在实际项目中，不应该将密码存储为明文，而是使用哈希算法进行存储和比对
        // 这里仅为演示，实际中应该使用更安全的方式
        return "admin".equals(username) && "21232f297a57a5a743894a0e4a801fc3".equals(md5(password));
    }

    private String md5(String input) {
        // 这里是一个简单的MD5加密示例，实际项目中应该使用更安全的加密方式
        // 注意：MD5不是安全的加密算法，仅作为演示使用
        // 实际项目中应该使用更安全的哈希算法，如BCrypt
        // 可以使用Spring Security等库来进行密码的安全处理
        return org.apache.commons.codec.digest.DigestUtils.md5Hex(input);
    }
}

```

![image](https://github.com/user-attachments/assets/0f2a0ae0-76e5-4565-a41a-1facc321e022)

## 万能工具：生成提示的提示
![image](https://github.com/user-attachments/assets/e5bca8a7-67e0-4695-a3d1-32d70c6ea1a1)

## 写给软件工程师的话
![image](https://github.com/user-attachments/assets/f6a44b6e-ad40-4518-92a3-bb36c7dde8d7)

## 小结
![image](https://github.com/user-attachments/assets/678c489b-d2ef-4a79-a2bd-c57126cee709)

## 思考题
请你思考一下，我们为什么要为大模型指定角色呢？欢迎你把你思考后的结果分享到评论区，也欢迎你把这节课的内容分享给其他朋友，我们下节课再见！

# 第一章：小试牛刀，理解基础概念 (3讲) 03｜探索智能体世界：LangChain与RAG检索增强生成
# 第二章：超燃实战，深度玩转 AI 模型 (4讲)
# 第二章：超燃实战，深度玩转 AI 模型 (4讲)
# 第二章：超燃实战，深度玩转 AI 模型 (4讲)
# 第二章：超燃实战，深度玩转 AI 模型 (4讲)
# 第三章：打入核心，挑战底层技术原理 (8讲)
# 第三章：打入核心，挑战底层技术原理 (8讲)
# 第三章：打入核心，挑战底层技术原理 (8讲)
# 第三章：打入核心，挑战底层技术原理 (8讲)
# 第三章：打入核心，挑战底层技术原理 (8讲)
# 第三章：打入核心，挑战底层技术原理 (8讲)
# 第三章：打入核心，挑战底层技术原理 (8讲)
# 第三章：打入核心，挑战底层技术原理 (8讲)
# 第四章：终极玩法，从0到1构建大模型 (10讲)
# 第四章：终极玩法，从0到1构建大模型 (10讲)
# 第四章：终极玩法，从0到1构建大模型 (10讲)
# 第四章：终极玩法，从0到1构建大模型 (10讲)
# 第四章：终极玩法，从0到1构建大模型 (10讲)
# 第四章：终极玩法，从0到1构建大模型 (10讲)
# 第四章：终极玩法，从0到1构建大模型 (10讲)
# 第四章：终极玩法，从0到1构建大模型 (10讲)
# 第四章：终极玩法，从0到1构建大模型 (10讲)
# 第四章：终极玩法，从0到1构建大模型 (10讲)
# 第五章：热点速递，AI行业发展趋势解读 (5讲)
# 第五章：热点速递，AI行业发展趋势解读 (5讲)
# 第五章：热点速递，AI行业发展趋势解读 (5讲)
# 第五章：热点速递，AI行业发展趋势解读 (5讲)
# 第五章：热点速递，AI行业发展趋势解读 (5讲)
# 结束语 (2讲)
# 结束语 (2讲)
