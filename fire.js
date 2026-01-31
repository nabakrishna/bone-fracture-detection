type RuleAction = "ALLOW" | "DENY";

interface FirewallRule {
  ip?: string;        // optional source IP
  port?: number;      // optional port
  action: RuleAction;
}

class Firewall {
  private rules: FirewallRule[] = [];

  addRule(rule: FirewallRule) {
    this.rules.push(rule);
  }

  checkTraffic(ip: string, port: number): RuleAction {
    for (const rule of this.rules) {
      const ipMatch = rule.ip ? rule.ip === ip : true;
      const portMatch = rule.port ? rule.port === port : true;

      if (ipMatch && portMatch) {
        return rule.action;
      }
    }
    return "DENY"; // default policy
  }
}

// ---- Usage ----
const firewall = new Firewall();

// allow HTTP & HTTPS
firewall.addRule({ port: 80, action: "ALLOW" });
firewall.addRule({ port: 443, action: "ALLOW" });

// block a specific IP
firewall.addRule({ ip: "192.168.1.10", action: "DENY" });

console.log(firewall.checkTraffic("192.168.1.5", 80));   // ALLOW
console.log(firewall.checkTraffic("192.168.1.10", 443)); // DENY
console.log(firewall.checkTraffic("10.0.0.1", 22));      // DENY
