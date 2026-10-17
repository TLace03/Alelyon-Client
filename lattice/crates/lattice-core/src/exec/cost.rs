//! Commands that may cost money: a curated list of the commands that create,
//! deploy or scale billed resources, or call metered paid services, so the
//! reader sees it before a command runs, and so allow all still asks for them
//! (spending money always asks; `convo::allow`).
//!
//! A rule names a program and the words that must follow it, in order (other
//! words may sit between them), for example `gcloud` then `compute instances
//! create`. It is a DECLARED list (as of [`AS_OF`]), matched on the command's
//! words: a command it does not know may still cost money, and one it names
//! may fall within a free tier. It never decides that a command is free.

/// When the list was written.
pub const AS_OF: &str = "2026-10-14";

/// Why a command may cost money.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cost {
    /// The service, in the reader's words.
    pub service: &'static str,
    /// What it does that may be billed.
    pub why: &'static str,
}

impl Cost {
    /// One line for the reader and the agent.
    pub fn line(&self) -> String {
        format!("May cost money: {} ({}).", self.why, self.service)
    }
}

/// (program, words that must follow in order, service, why).
type Rule = (&'static str, &'static [&'static str], &'static str, &'static str);

const RULES: &[Rule] = &[
    // Google Cloud
    ("gcloud", &["compute", "instances", "create"], "Google Cloud", "creates a virtual machine, billed while it exists"),
    ("gcloud", &["container", "clusters", "create"], "Google Cloud", "creates a Kubernetes cluster, billed while it runs"),
    ("gcloud", &["sql", "instances", "create"], "Google Cloud", "creates a Cloud SQL database, billed while it exists"),
    ("gcloud", &["run", "deploy"], "Google Cloud", "deploys a Cloud Run service, billed by use beyond the free tier"),
    ("gcloud", &["functions", "deploy"], "Google Cloud", "deploys a Cloud Function, billed by use beyond the free tier"),
    ("gcloud", &["app", "deploy"], "Google Cloud", "deploys to App Engine, billed by instance time"),
    ("gcloud", &["redis", "instances", "create"], "Google Cloud", "creates a Memorystore instance, billed while it exists"),
    ("gcloud", &["compute", "disks", "create"], "Google Cloud", "creates a disk, billed while it exists"),
    ("gcloud", &["services", "enable"], "Google Cloud", "turns on an API whose use may be billed"),
    ("gsutil", &["mb"], "Google Cloud", "creates a storage bucket, billed for what it stores"),
    // Amazon Web Services
    ("aws", &["ec2", "run-instances"], "AWS", "starts EC2 instances, billed while they run"),
    ("aws", &["rds", "create-db-instance"], "AWS", "creates an RDS database, billed while it exists"),
    ("aws", &["eks", "create-cluster"], "AWS", "creates an EKS cluster, billed while it runs"),
    ("aws", &["lambda", "create-function"], "AWS", "creates a Lambda function, billed by use beyond the free tier"),
    ("aws", &["cloudformation", "deploy"], "AWS", "deploys a stack whose resources may be billed"),
    ("aws", &["cloudformation", "create-stack"], "AWS", "creates a stack whose resources may be billed"),
    ("aws", &["elasticache", "create"], "AWS", "creates an ElastiCache cluster, billed while it exists"),
    ("aws", &["sagemaker", "create"], "AWS", "creates SageMaker resources, billed while they run"),
    ("aws", &["s3", "mb"], "AWS", "creates an S3 bucket, billed for what it stores"),
    ("aws", &["ecs", "create-service"], "AWS", "creates an ECS service, billed while its tasks run"),
    // Azure
    ("az", &["vm", "create"], "Azure", "creates a virtual machine, billed while it exists"),
    ("az", &["aks", "create"], "Azure", "creates a Kubernetes cluster, billed while it runs"),
    ("az", &["sql", "server", "create"], "Azure", "creates a SQL server, billed while it exists"),
    ("az", &["webapp", "create"], "Azure", "creates an App Service app on a billed plan"),
    ("az", &["appservice", "plan", "create"], "Azure", "creates an App Service plan, billed while it exists"),
    ("az", &["deployment", "create"], "Azure", "deploys a template whose resources may be billed"),
    ("az", &["deployment", "group", "create"], "Azure", "deploys a template whose resources may be billed"),
    // Infrastructure as code
    ("terraform", &["apply"], "Terraform", "creates or changes the infrastructure it describes, which may be billed"),
    ("tofu", &["apply"], "OpenTofu", "creates or changes the infrastructure it describes, which may be billed"),
    ("pulumi", &["up"], "Pulumi", "creates or changes the infrastructure it describes, which may be billed"),
    ("cdk", &["deploy"], "AWS CDK", "deploys stacks whose resources may be billed"),
    ("sam", &["deploy"], "AWS SAM", "deploys an application whose resources may be billed"),
    ("serverless", &["deploy"], "Serverless Framework", "deploys functions and resources that may be billed"),
    ("sls", &["deploy"], "Serverless Framework", "deploys functions and resources that may be billed"),
    // Platforms
    ("fly", &["deploy"], "Fly.io", "deploys machines, billed while they run"),
    ("fly", &["launch"], "Fly.io", "creates an app and its machines, billed while they run"),
    ("fly", &["scale"], "Fly.io", "changes how many or how large machines run, which changes the bill"),
    ("fly", &["volumes", "create"], "Fly.io", "creates a volume, billed while it exists"),
    ("flyctl", &["deploy"], "Fly.io", "deploys machines, billed while they run"),
    ("heroku", &["create"], "Heroku", "creates an app on a billed plan"),
    ("heroku", &["addons:create"], "Heroku", "adds a paid add-on"),
    ("heroku", &["ps:scale"], "Heroku", "changes the dynos that run, which changes the bill"),
    ("vercel", &["--prod"], "Vercel", "deploys to production; use beyond the plan is billed"),
    ("netlify", &["deploy", "--prod"], "Netlify", "deploys to production; use beyond the plan is billed"),
    ("railway", &["up"], "Railway", "deploys a service, billed by use"),
    ("doctl", &["create"], "DigitalOcean", "creates a resource, billed while it exists"),
    ("kubectl", &["scale"], "Kubernetes", "changes how many replicas run, which may change the bill"),
    // Paid compute and model services
    ("modal", &["deploy"], "Modal", "deploys functions billed by compute time"),
    ("modal", &["run"], "Modal", "runs a function billed by compute time"),
    ("runpodctl", &["create"], "RunPod", "creates a GPU pod, billed while it runs"),
    ("replicate", &["run"], "Replicate", "runs a model billed per prediction"),
];

/// Paid APIs a request tool (`curl`, `Invoke-RestMethod`, `Invoke-WebRequest`) may call: (host, service).
const PAID_HOSTS: &[(&str, &str)] = &[
    ("api.openai.com", "OpenAI's API"),
    ("api.anthropic.com", "Anthropic's API"),
    ("generativelanguage.googleapis.com", "the Gemini API"),
    ("api.mistral.ai", "Mistral's API"),
    ("api.together.xyz", "Together AI"),
    ("api.replicate.com", "Replicate"),
    ("api.stripe.com", "Stripe"),
    ("api.twilio.com", "Twilio"),
    ("api.sendgrid.com", "SendGrid"),
];

const REQUESTERS: &[&str] = &["curl", "curl.exe", "invoke-restmethod", "invoke-webrequest", "irm", "iwr", "wget", "http", "xh"];

/// The words of a command, lower case, split on spaces, quotes and the
/// separators PowerShell puts between commands (`;`, `|`, `&&`).
fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .replace("&&", " && ")
        .replace("||", " || ")
        .replace(';', " ; ")
        .replace('|', " | ")
        .replace(" |  | ", " || ")
        .split(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '`'))
        .filter(|w| !w.is_empty())
        .map(str::to_owned)
        .collect()
}

/// The program a word names: its file name without `.exe` or `.cmd`.
fn program(word: &str) -> &str {
    let name = word.rsplit(['/', '\\']).next().unwrap_or(word);
    name.strip_suffix(".exe").or_else(|| name.strip_suffix(".cmd")).unwrap_or(name)
}

/// Whether `rest` holds `want` in order, before the next command begins.
fn follows(rest: &[String], want: &[&str]) -> bool {
    let mut at = 0;
    for word in rest {
        if matches!(word.as_str(), ";" | "|" | "&&" | "||") {
            break;
        }
        if at < want.len() && word == want[at] {
            at += 1;
        }
    }
    at == want.len()
}

/// Why `text` may cost money, when the list knows; `None` when it does not
/// (which does not make it free).
pub fn cost_of(text: &str) -> Option<Cost> {
    let words = words(text);
    for (n, word) in words.iter().enumerate() {
        let name = program(word.trim_start_matches(['(', '&']));
        let rest = &words[n + 1..];
        for (prog, want, service, why) in RULES {
            if name == *prog && follows(rest, want) {
                return Some(Cost { service, why });
            }
        }
        if REQUESTERS.contains(&name)
            && let Some((_, service)) = rest
                .iter()
                .take_while(|w| !matches!(w.as_str(), ";" | "|" | "&&" | "||"))
                .find_map(|w| PAID_HOSTS.iter().find(|(host, _)| w.contains(host)))
        {
            return Some(Cost { service, why: "calls a paid API, billed per request" });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn billed_commands_are_named_with_why_and_others_are_not_called_free() {
        let gcloud = cost_of("gcloud compute instances create web-1 --zone us-central1-a").unwrap();
        assert_eq!(gcloud.service, "Google Cloud");
        assert!(gcloud.line().starts_with("May cost money: creates a virtual machine"));
        assert!(cost_of("terraform apply -auto-approve").is_some());
        assert!(cost_of("cd infra; terraform apply").is_some(), "a later command in the line counts");
        assert!(cost_of("C:\\tools\\aws.exe ec2 run-instances --image-id ami-1").is_some());
        assert!(cost_of("vercel --prod").is_some());
        assert!(cost_of("curl -s https://api.openai.com/v1/chat/completions -d @body.json").is_some());
        assert_eq!(cost_of("Invoke-RestMethod https://api.anthropic.com/v1/messages").unwrap().service, "Anthropic's API");
        // Reading, listing, planning and local work are not on the list.
        for free in ["gcloud compute instances list", "terraform plan", "aws s3 ls", "cargo build", "npm test", "curl https://example.com", "fly status"] {
            assert!(cost_of(free).is_none(), "{free}");
        }
        // The words must follow the program in one command, not across a separator.
        assert!(cost_of("terraform plan; echo apply").is_none());
    }
}
